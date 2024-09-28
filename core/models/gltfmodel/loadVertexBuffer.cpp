#include "gltfmodel.h"
#include "gltfutils.h"
#include "operations.h"

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
    math::Vector<float, 2> loadVec2(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::Vector<float, 2>(data[0], data[1]);
    }

    template <typename T = type>
    math::Vector<float, 3> loadVec3(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::Vector<float, 3>(loadVec2<T>(vertex), data[2]);
    }

    template <typename T = type>
    math::Vector<float, 4> loadVec4(size_t vertex) const {
        auto data = (const T*)buffer + vertex * stride;
        return math::Vector<float, 4>(loadVec3<T>(vertex), data[3]);
    }
};

template <typename type>
void pushBackIndex(const type* data, size_t count, uint32_t vertexStart, std::vector<uint32_t>& indexBuffer) {
    for (size_t index = 0; index < count; index++) {
        indexBuffer.push_back(data[index] + vertexStart);
    }
}

}

void GltfModel::loadVertexBuffer(const tinygltf::Model& gltfModel, const tinygltf::Node& node, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer) {
    for (const auto& children: node.children) {
        loadVertexBuffer(gltfModel, gltfModel.nodes[children], indexBuffer, vertexBuffer);
    }

    if (isInvalid(node.mesh)) return;

    const tinygltf::Mesh& mesh = gltfModel.meshes.at(node.mesh);
    for (const tinygltf::Primitive& primitive: mesh.primitives) {
        const auto pos = LoadBuffer<float>(primitive, gltfModel, "POSITION", TINYGLTF_TYPE_VEC3);
        const auto norm = LoadBuffer<float>(primitive, gltfModel, "NORMAL",TINYGLTF_TYPE_VEC3);
        const auto tex0 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_0", TINYGLTF_TYPE_VEC2);
        const auto tex1 = LoadBuffer<float>(primitive, gltfModel, "TEXCOORD_1", TINYGLTF_TYPE_VEC2);
        const auto weight = LoadBuffer<float>(primitive, gltfModel, "WEIGHTS_0",  TINYGLTF_TYPE_VEC4);
        const auto joint = LoadBuffer<void>(primitive, gltfModel, "JOINTS_0", TINYGLTF_TYPE_VEC4);

        uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
        uint32_t vertexCount = pos.accessor ? static_cast<uint32_t>(pos.accessor->count) : 0;
        for (uint32_t index = 0; index < vertexCount; index++) {
            interfaces::Vertex vert{};
            if (pos.buffer) vert.pos = pos.loadVec3(index);
            if (norm.buffer) vert.normal = math::normalize(norm.loadVec3(index));
            if (tex0.buffer) vert.uv0 = tex0.loadVec2(index);
            if (tex1.buffer) vert.uv1 = tex1.loadVec2(index);
            if (joint.buffer && joint.accessor) {
                switch (joint.accessor->componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        vert.joint0 = joint.loadVec4<uint16_t>(index);
                        break;
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        vert.joint0 = joint.loadVec4<uint8_t>(index);
                        break;
                }
            }
            if (weight.buffer) vert.weight0 = weight.loadVec4(index);
            vertexBuffer.push_back(vert);
        }

        if (isInvalid(primitive.indices)) continue;
        const GltfBufferExtractor indicesExtract(gltfModel, primitive.indices);

#define GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(ComponentType, type)                          \
    case ComponentType:                                                                     \
        pushBackIndex((const type*)indicesExtract.data, indicesExtract.count, vertexStart, indexBuffer);    \
        break;

        switch (indicesExtract.componentType) {
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT, uint32_t)
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT, uint16_t)
            GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE(TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE, uint8_t)
        }
#undef GLTFMODEL_LOADVERTEXBUFFER_INDEX_CASE
    }
}

}
