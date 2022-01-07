#include "gltfmodel.h"
#include "gltfutils.h"

#include <utils/operations.h>

#include <math/linearAlgebra.h>

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

void calculateTangent(uint32_t indexOffset, interfaces::Vertices& vertices, interfaces::Indices& indices) {
    for (uint32_t i = indexOffset; i < indices.size(); i += 3) {
        const auto& v0 = vertices[indices[i + 0]], & v1 = vertices[indices[i + 1]], & v2 = vertices[indices[i + 2]];

        const auto dv1 = v1.pos - v0.pos;
        const auto dv2 = v2.pos - v0.pos;
        const auto duv1 = v1.uv0 - v0.uv0;
        const auto duv2 = v2.uv0 - v0.uv0;

        const float det = 1.0f / (duv1[0] * duv2[1] - duv1[1] * duv2[0]);
        const auto bitangent = normalized(det * (duv1[0] * dv2 - duv2[0] * dv1));
        auto tangent = normalized(det * (duv2[1] * dv1 - duv1[1] * dv2));

        if (dot(cross(tangent, bitangent), v0.normal) < 0.0f) {
            tangent = -1.0f * tangent;
        }

        for (uint32_t j = i; j < i + 3; j++) {
            auto& v = vertices[indices[j]];
            v.tangent = normalized(tangent - v.normal * dot(v.normal, tangent));
        }
    }
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
            vertices.push_back(vert);
        }

        if (isInvalid(primitive.indices)) continue;
        const GltfBufferExtractor indicesExtract(gltfModel, primitive.indices);

        uint32_t indexOffset = indices.size();
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
        calculateTangent(indexOffset, vertices, indices);
    }
}

} // moon::models
