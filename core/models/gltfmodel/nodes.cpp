#include "gltfmodel.h"
#include "operations.h"

#include <numeric>

namespace moon::models {

namespace {
    moon::math::Matrix<float,4,4> localMatrix(Node* node){
        return translate(node->translation) * rotate(node->rotation) * scale(node->scale) * node->matrix;
    }

    moon::math::Matrix<float,4,4> getMatrix(Node* node) {
        return (node->parent ? getMatrix(node->parent) : moon::math::Matrix<float,4,4>(1.0f)) * localMatrix(node);
    }

    template <typename type>
    moon::math::Vector<float,4> loadJoint(const void* bufferJoints, int jointByteStride, size_t vertex){
        const type *buf = static_cast<const type*>(bufferJoints);
        return moon::math::Vector<float,4>(
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

Primitive::Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, moon::interfaces::Material* material, moon::interfaces::BoundingBox bb)
    : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material), bb(bb)
{}

Mesh::Mesh(VkPhysicalDevice physicalDevice, VkDevice device, moon::math::Matrix<float,4,4> matrix) {
    uniformBlock.matrix = matrix;
    uniformBuffer = utils::vkDefault::Buffer(physicalDevice, device, sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    moon::utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Mesh::Mesh, uniformBuffer");
};

void Node::update() {
    const auto matrix = getMatrix(this);
    mesh.uniformBlock.jointcount = skin ? std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS) : 0;
    mesh.uniformBlock.matrix = transpose(matrix);

    for (size_t i = 0; i < mesh.uniformBlock.jointcount; i++) {
        mesh.uniformBlock.jointMatrix[i] = transpose(inverse(matrix) * getMatrix(skin->joints[i]) * skin->inverseBindMatrices[i]);
    }
    mesh.uniformBuffer.copy(&mesh.uniformBlock);
}

void GltfModel::loadNode(Instance* instance, VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, uint32_t nodeIndex, const tinygltf::Model &model, uint32_t& indexStart) {
    Node& newNode = instance->nodes[nodeIndex];
    newNode.parent = parent;
    newNode.matrix = moon::math::Matrix<float,4,4>(1.0f);

    if (model.nodes[nodeIndex].translation.size() == 3) {
        newNode.translation = moon::math::Vector<float,3>(
            static_cast<float>(model.nodes[nodeIndex].translation[0]),
            static_cast<float>(model.nodes[nodeIndex].translation[1]),
            static_cast<float>(model.nodes[nodeIndex].translation[2])
        );
    }
    if (model.nodes[nodeIndex].rotation.size() == 4) {
        newNode.rotation = moon::math::Quaternion<float>(
            static_cast<float>(model.nodes[nodeIndex].rotation[3]),
            static_cast<float>(model.nodes[nodeIndex].rotation[0]),
            static_cast<float>(model.nodes[nodeIndex].rotation[1]),
            static_cast<float>(model.nodes[nodeIndex].rotation[2])
        );
    }
    if (model.nodes[nodeIndex].scale.size() == 3) {
        newNode.scale = moon::math::Vector<float,3>(
            static_cast<float>(model.nodes[nodeIndex].scale[0]),
            static_cast<float>(model.nodes[nodeIndex].scale[1]),
            static_cast<float>(model.nodes[nodeIndex].scale[2])
        );
    }
    if (model.nodes[nodeIndex].matrix.size() == 16) {
        const double* m = model.nodes[nodeIndex].matrix.data();
        newNode.matrix = moon::math::Matrix<float,4,4>(0.0f);
        for(uint32_t i = 0; i < 4; i++){
            for(uint32_t j = 0; j < 4; j++)
            newNode.matrix[i][j] = static_cast<float>(m[4*i + j]);
        }
    }

    for (const auto& children: model.nodes[nodeIndex].children) {
        loadNode(instance, physicalDevice, device, &newNode, children, model, indexStart);
    }

    if (model.nodes[nodeIndex].mesh > -1) {
        newNode.mesh = Mesh(physicalDevice, device, newNode.matrix);
        for (const tinygltf::Primitive &primitive: model.meshes[model.nodes[nodeIndex].mesh].primitives) {
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];

            uint32_t indexCount = primitive.indices > -1 ? model.accessors[primitive.indices > -1 ? primitive.indices : 0].count : 0;
            uint32_t vertexCount = posAccessor.count;

            newNode.mesh.primitives.emplace_back(
                Primitive(indexStart, indexCount, vertexCount, primitive.material > -1 ? &materials[primitive.material] : &materials.back(),
                    moon::interfaces::BoundingBox(
                        moon::math::Vector<float,3>(static_cast<float>(posAccessor.minValues[0]), static_cast<float>(posAccessor.minValues[1]), static_cast<float>(posAccessor.minValues[2])),
                        moon::math::Vector<float,3>(static_cast<float>(posAccessor.maxValues[0]), static_cast<float>(posAccessor.maxValues[1]), static_cast<float>(posAccessor.maxValues[2]))
                    )
                )
            );

            if (primitive.indices > -1){
                indexStart += static_cast<uint32_t>(model.accessors[primitive.indices > -1 ? primitive.indices : 0].count);
            }
        }
    }
}

void GltfModel::loadVertexBuffer(const tinygltf::Node& node, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer)
{
    for (const auto& children: node.children) {
        loadVertexBuffer(model.nodes[children], model, indexBuffer, vertexBuffer);
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
            vert.pos = pos.first ? moon::math::Vector<float,3>(pos.first[index * pos.second + 0], pos.first[index * pos.second + 1], pos.first[index * pos.second + 2]) : moon::math::Vector<float,3>(0.0f);
            vert.normal = normalize(moon::math::Vector<float,3>(normals.first ? moon::math::Vector<float,3>(normals.first[index * normals.second], normals.first[index * normals.second + 1], normals.first[index * normals.second + 2]) : moon::math::Vector<float,3>(0.0f)));
            vert.uv0 = texCoordSet0.first ? moon::math::Vector<float,2>(texCoordSet0.first[index * texCoordSet0.second], texCoordSet0.first[index * texCoordSet0.second + 1]) : moon::math::Vector<float,2>(0.0f);
            vert.uv1 = texCoordSet1.first ? moon::math::Vector<float,2>(texCoordSet1.first[index * texCoordSet1.second], texCoordSet1.first[index * texCoordSet1.second + 1]) : moon::math::Vector<float,2>(0.0f);
            vert.joint0 = moon::math::Vector<float,4>(0.0f, 0.0f, 0.0f, 0.0f);
            vert.weight0 = moon::math::Vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);

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

                vert.weight0 = moon::math::Vector<float,4>(
                    weights.first[index * weights.second + 0],
                    weights.first[index * weights.second + 1],
                    weights.first[index * weights.second + 2],
                    weights.first[index * weights.second + 3]
                );
                if (dot(vert.weight0,vert.weight0) == 0.0f) {
                    vert.weight0 = moon::math::Vector<float,4>(1.0f, 0.0f, 0.0f, 0.0f);
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
