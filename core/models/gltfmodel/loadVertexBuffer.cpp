#include "gltfmodel.h"
#include "operations.h"

namespace moon::models {

namespace {
    template <typename type>
    math::Vector<float,4> loadJoint(const void* bufferJoints, int jointByteStride, size_t vertex){
        const type *buf = static_cast<const type*>(bufferJoints);
        return math::Vector<float,4>(
            buf[vertex * jointByteStride + 0],
            buf[vertex * jointByteStride + 1],
            buf[vertex * jointByteStride + 2],
            buf[vertex * jointByteStride + 3]
        );
    }

    template <typename type>
    void pushBackIndex(const void *dataPtr, size_t count, uint32_t vertexStart, std::vector<uint32_t>& indexBuffer){
        const type *buf = static_cast<const type*>(dataPtr);
        for (size_t index = 0; index < count; index++) {
            indexBuffer.push_back(buf[index] + vertexStart);
        }
    }

    template <typename type>
    std::pair<const type*, int> loadBuffer(const tinygltf::Primitive& primitive, const tinygltf::Model& model, std::string attribute, uint64_t size, int TINYGLTF_TYPE){
        std::pair<const type*, int> buffer;
        if (primitive.attributes.find(attribute) != primitive.attributes.end()) {
            const tinygltf::Accessor &accessor = model.accessors[primitive.attributes.find(attribute)->second];
            const tinygltf::BufferView &view = model.bufferViews[accessor.bufferView];
            buffer.first = reinterpret_cast<const float *>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
            buffer.second = accessor.ByteStride(view) ? (accessor.ByteStride(view) / static_cast<int>(size)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE);
        }
        return buffer;
    }
}

void GltfModel::loadVertexBuffer(const tinygltf::Model& model, const tinygltf::Node& node, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer)
{
    for (const auto& children: node.children) {
        loadVertexBuffer(model, model.nodes[children], indexBuffer, vertexBuffer);
    }

    if (node.mesh <= -1) return;

    const tinygltf::Mesh& mesh = model.meshes[node.mesh];
    for (const tinygltf::Primitive& primitive: mesh.primitives)
    {
        const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];

        uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
        uint32_t vertexCount = static_cast<uint32_t>(posAccessor.count);

        std::pair<const float*, int> pos = loadBuffer<float>(primitive, model, "POSITION", sizeof(float), TINYGLTF_TYPE_VEC3);
        std::pair<const float*, int> normals = loadBuffer<float>(primitive, model, "NORMAL", sizeof(float), TINYGLTF_TYPE_VEC3);
        std::pair<const float*, int> texCoordSet0 = loadBuffer<float>(primitive, model, "TEXCOORD_0", sizeof(float), TINYGLTF_TYPE_VEC2);
        std::pair<const float*, int> texCoordSet1 = loadBuffer<float>(primitive, model, "TEXCOORD_1", sizeof(float), TINYGLTF_TYPE_VEC2);
        std::pair<const void*, int> joints = { nullptr, 0 };
        if (auto jointsAttr = primitive.attributes.find("JOINTS_0"); jointsAttr != primitive.attributes.end()) {
            joints = loadBuffer<void>(primitive, model, "JOINTS_0", tinygltf::GetComponentSizeInBytes(model.accessors[jointsAttr->second].componentType), TINYGLTF_TYPE_VEC4);
        }
        std::pair<const float*, int> weights = loadBuffer<float>(primitive, model, "WEIGHTS_0", sizeof(float), TINYGLTF_TYPE_VEC4);

        for (uint32_t index = 0; index < vertexCount; index++) {
            interfaces::Vertex vert{};
            vert.pos = pos.first ? math::Vector<float,3>(pos.first[index * pos.second + 0], pos.first[index * pos.second + 1], pos.first[index * pos.second + 2]) : math::Vector<float,3>(0.0f);
            vert.normal = normalize(math::Vector<float,3>(normals.first ? math::Vector<float,3>(normals.first[index * normals.second], normals.first[index * normals.second + 1], normals.first[index * normals.second + 2]) : math::Vector<float,3>(0.0f)));
            vert.uv0 = texCoordSet0.first ? math::Vector<float,2>(texCoordSet0.first[index * texCoordSet0.second], texCoordSet0.first[index * texCoordSet0.second + 1]) : math::Vector<float,2>(0.0f);
            vert.uv1 = texCoordSet1.first ? math::Vector<float,2>(texCoordSet1.first[index * texCoordSet1.second], texCoordSet1.first[index * texCoordSet1.second + 1]) : math::Vector<float,2>(0.0f);
            vert.joint0 = math::Vector<float,4>(0.0f, 0.0f, 0.0f, 0.0f);
            vert.weight0 = math::Vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);

            if (joints.first && weights.first)
            {
                switch (model.accessors[primitive.attributes.find("JOINTS_0")->second].componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        vert.joint0 = loadJoint<uint16_t>(joints.first, joints.second, index);
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        vert.joint0 = loadJoint<uint8_t>(joints.first, joints.second, index);
                        break;
                    }
                }

                vert.weight0 = math::Vector<float,4>(
                    weights.first[index * weights.second + 0],
                    weights.first[index * weights.second + 1],
                    weights.first[index * weights.second + 2],
                    weights.first[index * weights.second + 3]
                );
                if (dot(vert.weight0, vert.weight0) == 0.0f) {
                    vert.weight0 = math::Vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            vertexBuffer.push_back(vert);
        }

        if (primitive.indices <= -1) continue;

        const tinygltf::Accessor& accessor = model.accessors[primitive.indices > -1 ? primitive.indices : 0];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const void* dataPtr = &(model.buffers[bufferView.buffer].data[accessor.byteOffset + bufferView.byteOffset]);

        switch (accessor.componentType) {
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                pushBackIndex<uint32_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                break;
            }
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                pushBackIndex<uint16_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                break;
            }
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                pushBackIndex<uint8_t>(dataPtr, accessor.count, vertexStart, indexBuffer);
                break;
            }
        }
    }
}

}
