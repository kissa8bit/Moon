#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include <cstring>

#include "operations.h"
#include "device.h"

#include "gltfmodel.h"
#include "gltfutils.h"
#include "node.h"

namespace moon::models {

namespace {

void calculateTangent(std::vector<interfaces::Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer){
    for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
        const auto &v0 = vertexBuffer[indexBuffer[i + 0]], &v1 = vertexBuffer[indexBuffer[i + 1]], &v2 = vertexBuffer[indexBuffer[i + 2]];

        const auto dv1 = v1.pos - v0.pos;
        const auto dv2 = v2.pos - v0.pos;
        const auto duv1 = v1.uv0 - v0.uv0;
        const auto duv2 = v2.uv0 - v0.uv0;

        const float det = 1.0f / (duv1[0] * duv2[1] - duv1[1] * duv2[0]);
        const auto bitangent = normalized( det * (duv1[0] * dv2 - duv2[0] * dv1));
        auto tangent = normalized(det * (duv2[1] * dv1 - duv1[1] * dv2));

        if(dot(cross(tangent, bitangent), v0.normal) < 0.0f){
            tangent = -1.0f * tangent;
        }

        for(uint32_t j = i; j < i + 3; j++){
            auto& v = vertexBuffer[indexBuffer[j]];
            v.tangent = normalized(tangent - v.normal * dot(v.normal, tangent));
            v.bitangent = normalized(cross(v.normal, v.tangent));
        }
    }
}

}

GltfModel::GltfModel(std::filesystem::path filename, uint32_t instanceCount) : filename(filename) {
    instances.resize(instanceCount);
}

void GltfModel::destroyCache() {
    for(auto& texture: textures) texture.destroyCache();
    vertexCache = utils::Buffer();
    indexCache = utils::Buffer();
}

bool GltfModel::loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string error{}, warning{};
    const auto loadFileMethod = filename.extension() == ".glb" ? &tinygltf::TinyGLTF::LoadBinaryFromFile : &tinygltf::TinyGLTF::LoadASCIIFromFile;
    const auto loadSuccess = (gltfContext.*loadFileMethod)(&gltfModel, &error, &warning, filename.string(), tinygltf::SectionCheck::REQUIRE_VERSION);

    if (!CHECK_M(loadSuccess, "[ GltfModel::loadFromFile ] fail to load file")) return loadSuccess;

    loadTextures(gltfModel, device, commandBuffer);
    loadMaterials(gltfModel);

    for(auto& instance: instances){
        for (const auto& nodeIndex: gltfModel.scenes[isValid(gltfModel.defaultScene) ? gltfModel.defaultScene : 0].nodes) {
            instance.rootNodes.push_back(instance.loadNode(gltfModel, nodeIndex, nullptr));
        }
    }

    uint32_t indexStart = 0;
    std::vector<uint32_t> indexBuffer;
    std::vector<interfaces::Vertex> vertexBuffer;
    for (NodeId nodeId = 0; nodeId < gltfModel.nodes.size(); nodeId++) {
        const auto& node = gltfModel.nodes[nodeId];
        if (const auto meshIndex = node.mesh; isValid(meshIndex)) {
            meshes[nodeId] = GltfMesh(gltfModel, materials, meshIndex, indexStart);
            loadVertexBuffer(gltfModel, node, indexBuffer, vertexBuffer);
        }
    }
    calculateTangent(vertexBuffer, indexBuffer);

    loadSkins(gltfModel);
    if (gltfModel.animations.size() > 0) {
        loadAnimations(gltfModel);
    }

    for(auto& instance : instances){
        for (const auto& [nodeId, mesh] : meshes) {
            auto& skeleton = instance.skeletons[nodeId];
            skeleton.skin = mesh.skin;
            const size_t jointCount = mesh.skin ? mesh.skin->size() : 0;
            const size_t bufferSize = sizeof(math::mat4) * (jointCount + 1);
            skeleton.deviceBuffer = utils::vkDefault::Buffer(device, device.device(), bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            moon::utils::Memory::instance().nameMemory(skeleton.deviceBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Mesh::Mesh, uniformBuffer");
        }

        for (auto& [id, node] : instance.nodes) {
            if (boxMap.find(id) == boxMap.end()) {} {
                boxMap[id] = math::box();
            }
        }
        updateRootNodes(instance.rootNodes);
        for (auto& [rootNode, skeleton] : instance.skeletons) {
            skeleton.update(instance.nodes, rootNode);
        }
    }

    for (const auto& vertex : vertexBuffer) {
        const auto& pos = vertex.pos;
        for (uint32_t i = 0; i < 4; i++) {
            auto& box = boxMap[i];
            box.max = math::max(box.max, pos);
            box.min = math::min(box.min, pos);
        }
    }

    utils::createDeviceBuffer(device, device.device(), commandBuffer, vertexBuffer.size() * sizeof(interfaces::Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, vertices);
    if (!indexBuffer.empty()) {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);
    }
    return true;
}

void GltfModel::createDescriptors(const utils::PhysicalDevice& device) {
    meshDescriptorSetLayout = interfaces::Model::createMeshDescriptorSetLayout(device.device());
    materialDescriptorSetLayout = interfaces::Model::createMaterialDescriptorSetLayout(device.device());

    std::vector<const utils::vkDefault::DescriptorSetLayout*> layouts(materials.size(), &materialDescriptorSetLayout);
    for (const auto& instance : instances) {
        for (const auto& _ : instance.nodes) {
            layouts.push_back(&meshDescriptorSetLayout);
        }
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), layouts, 1);

    for (auto& instance : instances) {
        for (auto& [nodeId, skeleton] : instance.skeletons) {
            skeleton.createDescriptorSet(device.device(), descriptorPool, meshDescriptorSetLayout);
        }
    }

    for (auto& material : materials) {
        material.createDescriptorSet(device.device(), descriptorPool, materialDescriptorSetLayout);
    }
}

void GltfModel::create(const utils::PhysicalDevice& device, VkCommandPool commandPool) {
    if(
        CHECK_M(VkPhysicalDevice(device), "[ GltfModel::create ] VkPhysicalDevice is VK_NULL_HANDLE") &&
        CHECK_M(VkDevice(device.device()), "[ GltfModel::create ] VkDevice is VK_NULL_HANDLE") &&
        CHECK_M(commandPool, "[ GltfModel::create ] VkCommandPool is VK_NULL_HANDLE")
    ) {
        utils::singleCommandBuffer::Scoped commandBuffer(device.device(), device.device()(0, 0), commandPool);
        if (loadFromFile(device, commandBuffer)) {
            createDescriptors(device);
        }
    }
    destroyCache();
}

void GltfModel::render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t &primitiveCount) const {
    VkDeviceSize offsets = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertices, &offsets);
    if (VkBuffer(indices) != VK_NULL_HANDLE) {
        vkCmdBindIndexBuffer(commandBuffer, indices, 0, VK_INDEX_TYPE_UINT32);
    }

    const auto& instance = instances.at(instanceNumber);
    for (const auto& [nodeId, mesh] : meshes) {
        const auto& skeleton = instance.skeletons.at(nodeId);
        auto descriptors = descriptorSets;
        descriptors.push_back(skeleton.descriptorSet);
        mesh.render(commandBuffer, pipelineLayout, descriptors, primitiveCount);
    }
}

void GltfModel::renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    for (auto& [_, node] : instances.at(instanceNumber).nodes) {
        if (!CHECK_M(node.get(), std::string("[ GltfModel::render ] node is nullptr"))) continue;
        // node->mesh.renderBB(commandBuffer, pipelineLayout, descriptorSets);
    }
}

std::vector<interfaces::Animation*> GltfModel::animations(uint32_t instanceNumber) {
    auto& animations = instances[instanceNumber].animations;
    std::vector<interfaces::Animation*> animationsPtrs;
    animationsPtrs.reserve(animations.size());
    for(auto& animation: animations){
        animationsPtrs.emplace_back(&animation);
    }
    return animationsPtrs;
}

}
