#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include <cstring>

#include <utils/operations.h>
#include <utils/device.h>

#include "gltfmodel.h"
#include "gltfutils.h"
#include "gltfskeleton.h"
#include "node.h"
#include "loadVertices.h"


namespace moon::models {

GltfModel::GltfModel(std::filesystem::path filename, uint32_t instanceCount) :
    filename(filename)
{
    type = interfaces::Model::VertexType::base;

    instances.resize(instanceCount);
}

void GltfModel::destroyCache() {
    for(auto& [_, texture] : textures) texture.destroyCache();
    cache = Cache();
}

bool GltfModel::loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string error{}, warning{};
    const auto loadFileMethod = filename.extension() == ".glb" ? &tinygltf::TinyGLTF::LoadBinaryFromFile : &tinygltf::TinyGLTF::LoadASCIIFromFile;
    const auto loadSuccess = (gltfContext.*loadFileMethod)(&gltfModel, &error, &warning, filename.string(), tinygltf::SectionCheck::REQUIRE_VERSION);

    if (!CHECK_M(loadSuccess, "[ GltfModel::loadFromFile ] fail to load file")) return loadSuccess;

    loadMaterials(gltfModel, device, commandBuffer);

    struct {
        interfaces::Indices indices;
        interfaces::Vertices vertices;
    } host;

    uint32_t indexStart = 0;
    for (Node::Id nodeId = 0; nodeId < gltfModel.nodes.size(); nodeId++) {
        const auto& node = gltfModel.nodes[nodeId];

        const auto meshIndex = node.mesh;
        const bool isMeshValid = isValid(meshIndex);
        if (isMeshValid) {
            meshes[nodeId] = GltfMesh(gltfModel, gltfModel.meshes[meshIndex], materials, indexStart);
            loadVertices(gltfModel, gltfModel.meshes[meshIndex], host.indices, host.vertices);
        }

        const auto skinIndex = node.skin;
        const bool isSkinValid = isValid(skinIndex);
        if (isSkinValid) {
            skins[nodeId] = Skin(gltfModel, gltfModel.skins[skinIndex]);
            if (isMeshValid) {
                meshes.at(nodeId).calculateNodeBoxes(host.vertices, host.indices, skins.at(nodeId));
            }
        }

        for (auto& instance : instances) {
            if (instance.nodes.find(nodeId) == instance.nodes.end()) {
                instance.loadNode(gltfModel, nodeId, nullptr);
            }
            if (isMeshValid) {
                instance.skeletons[nodeId] = GltfSkeleton(device, isSkinValid ? &skins.at(nodeId) : nullptr);
            }
        }
    }

    utils::createDeviceBuffer(device, device.device(), commandBuffer, host.vertices.size() * sizeof(interfaces::Vertex), host.vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cache.vertices, vertices);
    if (!host.indices.empty()) {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, host.indices.size() * sizeof(uint32_t), host.indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, cache.indices, indices);
    }

    loadAnimations(gltfModel);

    for (auto& instance : instances) {
        updateNodes(instance.nodes, instance.skeletons);
    }

    return true;
}

void GltfModel::createDescriptors(const utils::PhysicalDevice& device) {
    skeletonDescriptorSetLayout = interfaces::Skeleton::descriptorSetLayout(device.device());
    materialDescriptorSetLayout = interfaces::Material::descriptorSetLayout(device.device());

    std::vector<const utils::vkDefault::DescriptorSetLayout*> layouts(materials.size(), &materialDescriptorSetLayout);
    for (const auto& instance : instances) {
        for (const auto& _ : instance.nodes) {
            layouts.push_back(&skeletonDescriptorSetLayout);
        }
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), layouts, 1);

    for (auto& instance : instances) {
        for (auto& [nodeId, skeleton] : instance.skeletons) {
            skeleton.createDescriptorSet(device.device(), descriptorPool, skeletonDescriptorSetLayout);
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
    const auto& instance = instances.at(instanceNumber);
    for (const auto& [nodeId, mesh] : meshes) {
        const auto& skeleton = instance.skeletons.at(nodeId);
        auto descriptors = descriptorSets;
        descriptors.push_back(skeleton.descriptorSet);

        const auto renderBB = (skins.find(nodeId) == skins.end() ? &GltfMesh::renderBB : &GltfMesh::renderNodeBB);
        (mesh.*renderBB)(commandBuffer, pipelineLayout, descriptors);
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
