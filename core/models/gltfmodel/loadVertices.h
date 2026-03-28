#include "gltfmodel.h"
#include "gltfutils.h"

#include <utils/operations.h>

#include <math/linearAlgebra.h>

#include <interfaces/model.h>

namespace moon::models {

namespace {

template <typename type>
struct LoadBuffer {
    const type* buffer{ nullptr };
    const tinygltf::Accessor* accessor{nullptr};
    int stride{0};

    LoadBuffer(const tinygltf::Primitive& primitive, const tinygltf::Model& gltfModel, const std::string& attribute, int TINYGLTF_TYPE) {
        if (const auto attributeIt = primitive.attributes.find(attribute); attributeIt != primitive.attributes.end()) {
            const auto& [_, attribute] = *attributeIt;
            accessor = &gltfModel.accessors.at(attribute);
            const auto& view = gltfModel.bufferViews.at(accessor->bufferView);
            const auto byteStride = accessor->ByteStride(view);
            const auto size = tinygltf::GetComponentSizeInBytes(accessor->componentType);
            buffer = (const type*)&gltfModel.buffers[view.buffer].data[accessor->byteOffset + view.byteOffset];
            stride = byteStride ? byteStride / size : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE);
        }
    }

    LoadBuffer(const std::map<std::string, int>& attributeMap, const tinygltf::Model& gltfModel, const std::string& attribute, int TINYGLTF_TYPE) {
        if (const auto attributeIt = attributeMap.find(attribute); attributeIt != attributeMap.end()) {
            const auto& [_, accessorIndex] = *attributeIt;
            accessor = &gltfModel.accessors.at(accessorIndex);
            const auto& view = gltfModel.bufferViews.at(accessor->bufferView);
            const auto byteStride = accessor->ByteStride(view);
            const auto size = tinygltf::GetComponentSizeInBytes(accessor->componentType);
            buffer = (const type*)&gltfModel.buffers[view.buffer].data[accessor->byteOffset + view.byteOffset];
            stride = byteStride ? byteStride / size : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE);
        }
    }

    template <typename T = type>
    math::vec2 loadVec2(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::vec2(data[0], data[1]);
    }

    template <typename T = type>
    math::vec3 loadVec3(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::vec3(loadVec2<T>(vertex), data[2]);
    }

    template <typename T = type>
    math::vec4 loadVec4(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::vec4(loadVec3<T>(vertex), data[3]);
    }
};

template <typename type>
void pushBackIndex(const type* data, size_t count, uint32_t vertexStart, interfaces::Indices& indices) {
    for (size_t index = 0; index < count; index++) {
        indices.push_back(data[index] + vertexStart);
    }
}

template<typename VerticesT>
void calculateTangent(uint32_t indexOffset, VerticesT& vertices, interfaces::Indices& indices) {
    const uint32_t indexCount = static_cast<uint32_t>(indices.size()) - indexOffset;
    if (indexCount < 3) return;

    // Find vertex range for bitangent accumulation
    uint32_t minIdx = indices[indexOffset], maxIdx = indices[indexOffset];
    for (uint32_t i = 1; i < indexCount; i++) {
        uint32_t idx = indices[indexOffset + i];
        if (idx < minIdx) minIdx = idx;
        if (idx > maxIdx) maxIdx = idx;
    }

    std::vector<math::vec3> bitangents(maxIdx - minIdx + 1, math::vec3(0.0f));

    for (uint32_t i = 0; i < indexCount; i++) {
        vertices[indices[indexOffset + i]].tangent = math::vec4(0.0f);
    }

    // Accumulate per-triangle tangent and bitangent
    for (uint32_t i = 0; i < indexCount; i += 3) {
        uint32_t i0 = indices[indexOffset + i + 0];
        uint32_t i1 = indices[indexOffset + i + 1];
        uint32_t i2 = indices[indexOffset + i + 2];

        math::vec3 e1 = vertices[i1].pos - vertices[i0].pos;
        math::vec3 e2 = vertices[i2].pos - vertices[i0].pos;

        math::vec2 duv1 = vertices[i1].uv0 - vertices[i0].uv0;
        math::vec2 duv2 = vertices[i2].uv0 - vertices[i0].uv0;

        float det = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        if (det > -1e-6f && det < 1e-6f) continue;
        float invDet = 1.0f / det;

        math::vec3 t = (e1 * duv2[1] - e2 * duv1[1]) * invDet;
        math::vec3 b = (e2 * duv1[0] - e1 * duv2[0]) * invDet;

        for (uint32_t idx : {i0, i1, i2}) {
            auto& tang = vertices[idx].tangent;
            tang = math::vec4(math::vec3(tang[0], tang[1], tang[2]) + t, 0.0f);
            bitangents[idx - minIdx] = bitangents[idx - minIdx] + b;
        }
    }

    // Orthogonalize, normalize, compute handedness
    for (uint32_t i = 0; i < indexCount; i++) {
        uint32_t idx = indices[indexOffset + i];
        auto& vert = vertices[idx];
        math::vec3 t(vert.tangent[0], vert.tangent[1], vert.tangent[2]);
        math::vec3 n = vert.normal;

        t = t - n * math::dot(n, t);
        float len = t.norm();
        if (len < 1e-6f) continue;
        t = t * (1.0f / len);

        // Shader uses b = cross(t, n) * w, match handedness to accumulated bitangent
        float w = math::dot(math::cross(t, n), bitangents[idx - minIdx]) < 0.0f ? -1.0f : 1.0f;
        vert.tangent = math::vec4(t, w);
    }
}

// Loads a dense vec3 array for a morph target attribute, handling sparse accessors.
// If the accessor has bufferView == -1, base data is implicitly all zeros.
// Sparse overrides are applied on top.
std::vector<math::vec3> loadSparseMorphDeltaVec3(
    const tinygltf::Model& gltfModel,
    const std::map<std::string, int>& target,
    const std::string& attr,
    uint32_t vertexCount)
{
    std::vector<math::vec3> deltas(vertexCount, math::vec3(0.0f));

    const auto it = target.find(attr);
    if (it == target.end()) return deltas;

    const auto& accessor = gltfModel.accessors.at(it->second);

    if (accessor.bufferView >= 0) {
        const auto& view = gltfModel.bufferViews.at(accessor.bufferView);
        const auto* data = gltfModel.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset;
        const int byteStride = accessor.ByteStride(view);
        const int stride = byteStride ? byteStride / static_cast<int>(sizeof(float)) : 3;
        for (uint32_t v = 0; v < vertexCount; v++) {
            const float* ptr = reinterpret_cast<const float*>(data) + v * stride;
            deltas[v] = math::vec3(ptr[0], ptr[1], ptr[2]);
        }
    }

    if (accessor.sparse.isSparse) {
        const auto& si = accessor.sparse.indices;
        const auto& sv = accessor.sparse.values;
        const auto& idxView = gltfModel.bufferViews.at(si.bufferView);
        const auto& valView = gltfModel.bufferViews.at(sv.bufferView);
        const auto* idxData = gltfModel.buffers[idxView.buffer].data.data() + idxView.byteOffset + si.byteOffset;
        const auto* valData = reinterpret_cast<const float*>(gltfModel.buffers[valView.buffer].data.data() + valView.byteOffset + sv.byteOffset);
        for (int s = 0; s < accessor.sparse.count; s++) {
            uint32_t idx = 0;
            if (si.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)       idx = reinterpret_cast<const uint8_t*>(idxData)[s];
            else if (si.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)  idx = reinterpret_cast<const uint16_t*>(idxData)[s];
            else                                                                   idx = reinterpret_cast<const uint32_t*>(idxData)[s];
            deltas[idx] = math::vec3(valData[s * 3], valData[s * 3 + 1], valData[s * 3 + 2]);
        }
    }

    return deltas;
}

// Returns per-primitive SSBO data: {morphTargetCount, vertexCount, vertexStart, pad, posDeltas[], normDeltas[]}
std::vector<std::vector<uint8_t>> loadMorphDeltasForMesh(
    const tinygltf::Model& gltfModel,
    const tinygltf::Mesh& mesh,
    uint32_t meshVertexStart)
{
    std::vector<std::vector<uint8_t>> result;
    uint32_t primitiveVertexStart = meshVertexStart;

    for (const tinygltf::Primitive& primitive : mesh.primitives) {
        uint32_t vertexCount = 0;
        if (const auto it = primitive.attributes.find("POSITION"); it != primitive.attributes.end()) {
            vertexCount = static_cast<uint32_t>(gltfModel.accessors.at(it->second).count);
        }

        const uint32_t morphTargetCount = static_cast<uint32_t>(primitive.targets.size());

        using Header = interfaces::MorphDeltas::Header;

        if (morphTargetCount == 0 || vertexCount == 0) {
            std::vector<uint8_t> buf(sizeof(Header), 0);
            result.push_back(std::move(buf));
            primitiveVertexStart += vertexCount;
            continue;
        }

        const size_t deltaCount = morphTargetCount * vertexCount;
        const size_t bufSize = sizeof(Header) + deltaCount * 2 * sizeof(math::vec4);
        std::vector<uint8_t> buf(bufSize, 0);

        Header& header = *reinterpret_cast<Header*>(buf.data());
        header.morphTargetCount = morphTargetCount;
        header.vertexCount = vertexCount;
        header.vertexStart = primitiveVertexStart;
        header._pad = 0;

        math::vec4* posDeltas  = reinterpret_cast<math::vec4*>(buf.data() + sizeof(Header));
        math::vec4* normDeltas = posDeltas + deltaCount;

        for (uint32_t t = 0; t < morphTargetCount; t++) {
            const auto& target = primitive.targets[t];

            const auto pd = loadSparseMorphDeltaVec3(gltfModel, target, "POSITION", vertexCount);
            for (uint32_t v = 0; v < vertexCount; v++) {
                posDeltas[t * vertexCount + v] = math::vec4(pd[v], 0.0f);
            }

            const auto nd = loadSparseMorphDeltaVec3(gltfModel, target, "NORMAL", vertexCount);
            for (uint32_t v = 0; v < vertexCount; v++) {
                normDeltas[t * vertexCount + v] = math::vec4(nd[v], 0.0f);
            }
        }

        result.push_back(std::move(buf));
        primitiveVertexStart += vertexCount;
    }

    return result;
}

void loadPBRVertices(const tinygltf::Model& gltfModel, const tinygltf::Mesh& mesh, interfaces::Indices& indices, interfaces::PBRVertices& vertices) {
    for (const tinygltf::Primitive& primitive: mesh.primitives) {
        const auto pos  = LoadBuffer<float>(primitive, gltfModel, "POSITION",   TINYGLTF_TYPE_VEC3);
        const auto norm = LoadBuffer<float>(primitive, gltfModel, "NORMAL",     TINYGLTF_TYPE_VEC3);
        const auto tex0 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_0", TINYGLTF_TYPE_VEC2);
        const auto tex1 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_1", TINYGLTF_TYPE_VEC2);
        const auto tan  = LoadBuffer<float>(primitive, gltfModel, "TANGENT",    TINYGLTF_TYPE_VEC4);

        uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
        uint32_t vertexCount = pos.accessor ? static_cast<uint32_t>(pos.accessor->count) : 0;
        for (uint32_t index = 0; index < vertexCount; index++) {
            interfaces::PBRVertex vert{};
            if (pos.buffer)  vert.pos     = pos.loadVec3(index);
            if (norm.buffer) vert.normal  = math::normalized(norm.loadVec3(index));
            if (tex0.buffer) vert.uv0     = tex0.loadVec2(index);
            if (tex1.buffer) vert.uv1     = tex1.loadVec2(index);
            if (tan.buffer)  vert.tangent = tan.loadVec4(index);
            vertices.push_back(vert);
        }

        if (isInvalid(primitive.indices)) continue;
        const GltfBufferExtractor indicesExtract(gltfModel, primitive.indices);

        uint32_t indexOffset = static_cast<uint32_t>(indices.size());
#define GLTFMODEL_LOADPBRVERTEXBUFFER_INDEX_CASE(ComponentType, type)                                       \
    case ComponentType:                                                                                     \
        pushBackIndex((const type*)indicesExtract.data, indicesExtract.count, vertexStart, indices);        \
        break;

        switch (indicesExtract.componentType) {
            GLTFMODEL_LOADPBRVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT, uint32_t)
            GLTFMODEL_LOADPBRVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT, uint16_t)
            GLTFMODEL_LOADPBRVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE, uint8_t)
        }
#undef GLTFMODEL_LOADPBRVERTEXBUFFER_INDEX_CASE
        if (!tan.buffer) calculateTangent(indexOffset, vertices, indices);
    }
}

void loadVertices(const tinygltf::Model& gltfModel, const tinygltf::Mesh& mesh, interfaces::Indices& indices, interfaces::Vertices& vertices) {
    for (const tinygltf::Primitive& primitive: mesh.primitives) {
        const auto pos = LoadBuffer<float>(primitive, gltfModel, "POSITION", TINYGLTF_TYPE_VEC3);
        const auto norm = LoadBuffer<float>(primitive, gltfModel, "NORMAL",TINYGLTF_TYPE_VEC3);
        const auto tex0 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_0", TINYGLTF_TYPE_VEC2);
        const auto tex1 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_1", TINYGLTF_TYPE_VEC2);
        const auto weight = LoadBuffer<float>(primitive, gltfModel, "WEIGHTS_0",  TINYGLTF_TYPE_VEC4);
        const auto joint = LoadBuffer<void>(primitive, gltfModel, "JOINTS_0", TINYGLTF_TYPE_VEC4);
        const auto tan = LoadBuffer<float>(primitive, gltfModel, "TANGENT", TINYGLTF_TYPE_VEC4);

        uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
        uint32_t vertexCount = pos.accessor ? static_cast<uint32_t>(pos.accessor->count) : 0;
        for (uint32_t index = 0; index < vertexCount; index++) {
            interfaces::Vertex vert{};
            if (pos.buffer) vert.pos = pos.loadVec3(index);
            if (norm.buffer) vert.normal = math::normalized(norm.loadVec3(index));
            if (tex0.buffer) vert.uv0 = tex0.loadVec2(index);
            if (tex1.buffer) vert.uv1 = tex1.loadVec2(index);
            if (joint.buffer && joint.accessor) {
                switch (joint.accessor->componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        vert.joint = joint.loadVec4<uint16_t>(index);
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        vert.joint = joint.loadVec4<uint8_t>(index);
                        break;
                }
            }
            if (weight.buffer) vert.weight = weight.loadVec4(index);
            if (tan.buffer) vert.tangent = tan.loadVec4(index);
            vertices.push_back(vert);
        }

        if (isInvalid(primitive.indices)) continue;
        const GltfBufferExtractor indicesExtract(gltfModel, primitive.indices);

        uint32_t indexOffset = static_cast<uint32_t>(indices.size());
#define GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(ComponentType, type)                                          \
    case ComponentType:                                                                                     \
        pushBackIndex((const type*)indicesExtract.data, indicesExtract.count, vertexStart, indices);        \
        break;

        switch (indicesExtract.componentType) {
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT, uint32_t)
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT, uint16_t)
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE, uint8_t)
        }
#undef GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE
        if (!tan.buffer) calculateTangent(indexOffset, vertices, indices);
    }
}

} // anonymous namespace

} // moon::models
