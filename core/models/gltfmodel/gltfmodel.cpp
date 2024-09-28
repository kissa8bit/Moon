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
        const auto bitangent = normalize( det * (duv1[0] * dv2 - duv2[0] * dv1));
        auto tangent = normalize(det * (duv2[1] * dv1 - duv1[1] * dv2));

        if(dot(cross(tangent, bitangent), v0.normal) < 0.0f){
            tangent = -1.0f * tangent;
        }

        for(uint32_t j = i; j < i + 3; j++){
            auto& v = vertexBuffer[indexBuffer[j]];
            v.tangent = normalize(tangent - v.normal * dot(v.normal, tangent));
            v.bitangent = normalize(cross(v.normal, v.tangent));
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

void GltfModel::loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;

    std::string error{}, warning{};
    const auto loadFileMethod = filename.extension() == ".glb" ? &tinygltf::TinyGLTF::LoadBinaryFromFile : &tinygltf::TinyGLTF::LoadASCIIFromFile;
    const auto loadSuccess = (gltfContext.*loadFileMethod)(&gltfModel, &error, &warning, filename.string(), tinygltf::SectionCheck::REQUIRE_VERSION);

    if (!loadSuccess) return;

    loadTextures(gltfModel, device, commandBuffer);
    loadMaterials(gltfModel);

    for(auto& instance: instances){
        uint32_t indexStart = 0;
        for (const auto& nodeIndex: gltfModel.scenes[isValid(gltfModel.defaultScene) ? gltfModel.defaultScene : 0].nodes) {
            instance.rootNodes.push_back(loadNode(gltfModel, instance.nodes, nullptr, nodeIndex, indexStart));
        }
    }

    std::vector<uint32_t> indexBuffer;
    std::vector<interfaces::Vertex> vertexBuffer;
    for (const auto& nodeIndex: gltfModel.scenes[isValid(gltfModel.defaultScene) ? gltfModel.defaultScene : 0].nodes) {
        loadVertexBuffer(gltfModel, gltfModel.nodes[nodeIndex], indexBuffer, vertexBuffer);
    }
    calculateTangent(vertexBuffer, indexBuffer);

    loadSkins(gltfModel);
    if (gltfModel.animations.size() > 0) {
        loadAnimations(gltfModel);
    }

    for(auto& instance : instances){
        for (auto& [_, node] : instance.nodes) {
            node->mesh.createDeviceBuffer(device);
        }
        updateRootNodes(instance.rootNodes);
    }

    utils::createDeviceBuffer(device, device.device(), commandBuffer, vertexBuffer.size() * sizeof(interfaces::Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, vertices);
    if (!indexBuffer.empty()) {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);
    }
}

void GltfModel::createDescriptors(VkDevice device) {
    meshDescriptorSetLayout = interfaces::Model::createMeshDescriptorSetLayout(device);
    materialDescriptorSetLayout = interfaces::Model::createMaterialDescriptorSetLayout(device);

    std::vector<const utils::vkDefault::DescriptorSetLayout*> layouts(materials.size(), &materialDescriptorSetLayout);
    for (const auto& instance : instances) {
        for (const auto& _ : instance.nodes) {
            layouts.push_back(&meshDescriptorSetLayout);
        }
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device, layouts, 1);

    for (auto& instance : instances) {
        for (auto& [_, node] : instance.nodes) {
            node->mesh.createDescriptorSet(device, descriptorPool, meshDescriptorSetLayout);
        }
    }

    for (auto& material : materials) {
        material.createDescriptorSet(device, descriptorPool, materialDescriptorSetLayout);
    }
}

void GltfModel::create(const utils::PhysicalDevice& device, VkCommandPool commandPool) {
    if(
        CHECK_M(VkPhysicalDevice(device), std::string("[ GltfModel::create ] VkPhysicalDevice is VK_NULL_HANDLE")) &&
        CHECK_M(VkDevice(device.device()), std::string("[ GltfModel::create ] VkDevice is VK_NULL_HANDLE")) &&
        CHECK_M(commandPool, std::string("[ GltfModel::create ] VkCommandPool is VK_NULL_HANDLE"))
    ) {
        utils::singleCommandBuffer::Scoped commandBuffer(device.device(), device.device()(0, 0), commandPool);
        loadFromFile(device, commandBuffer);
        createDescriptors(device.device());
    }
    destroyCache();
}

void GltfModel::render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t &primitiveCount) const {
    for (auto& [_, node] : instances.at(instanceNumber).nodes) {
        if (!CHECK_M(node.get(), std::string("[ GltfModel::render ] node is nullptr"))) continue;
        node->mesh.render(commandBuffer, pipelineLayout, descriptorSets, primitiveCount);
    }
}

void GltfModel::renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    for (auto& [_, node] : instances.at(instanceNumber).nodes) {
        if (!CHECK_M(node.get(), std::string("[ GltfModel::render ] node is nullptr"))) continue;
        node->mesh.renderBB(commandBuffer, pipelineLayout, descriptorSets);
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
