#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include <cstring>

#include <utils/operations.h>
#include <utils/device.h>

#include "gltfmodel.h"
#include "gltfutils.h"
#include "gltfskeleton.h"
#include "gltfmorph.h"
#include "node.h"
#include "loadVertices.h"


namespace moon::models {

GltfModel::GltfModel(std::filesystem::path filename, uint32_t instanceCount) :
    filename(filename)
{
    type = interfaces::Model::VertexType::animated;

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

    type = (gltfModel.animations.empty() && gltfModel.skins.empty())
        ? interfaces::Model::VertexType::pbr
        : interfaces::Model::VertexType::animated;

    interfaces::Indices hostIndices;
    interfaces::Vertices hostVertices;
    interfaces::PBRVertices hostPBRVertices;

    uint32_t indexStart = 0;
    for (Node::Id nodeId = 0; nodeId < gltfModel.nodes.size(); nodeId++) {
        const auto& node = gltfModel.nodes[nodeId];

        for (auto& instance : instances) {
            if (instance.nodes.find(nodeId) == instance.nodes.end()) {
                instance.loadNode(gltfModel, nodeId, nullptr);
            }
        }

        if (!isValid(node.mesh)) continue;

        const auto& gltfMesh = gltfModel.meshes[node.mesh];
        const uint32_t meshVertexStart = type == interfaces::Model::VertexType::pbr
            ? static_cast<uint32_t>(hostPBRVertices.size())
            : static_cast<uint32_t>(hostVertices.size());
        const uint32_t morphTargetCount = gltfMesh.primitives.empty() ? 0 : static_cast<uint32_t>(gltfMesh.primitives[0].targets.size());
        const bool isSkinValid = isValid(node.skin);

        meshes[nodeId] = GltfMesh(gltfModel, gltfMesh, materials, indexStart);
        if (type == interfaces::Model::VertexType::pbr) {
            loadPBRVertices(gltfModel, gltfMesh, hostIndices, hostPBRVertices);
        } else {
            loadVertices(gltfModel, gltfMesh, hostIndices, hostVertices);
        }

        auto morphDeltaBuffers = loadMorphDeltasForMesh(gltfModel, gltfMesh, meshVertexStart);
        auto& mesh = meshes.at(nodeId);
        for (size_t pi = 0; pi < std::min(morphDeltaBuffers.size(), mesh.primitives.size()); pi++) {
            const auto& data = morphDeltaBuffers[pi];
            utils::Buffer stagingBuffer;
            utils::createDeviceBuffer(device, device.device(), commandBuffer, data.size(), data.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, stagingBuffer, mesh.primitives[pi].morphDeltas.deviceBuffer);
            cache.morphDelta.push_back(std::move(stagingBuffer));
        }

        if (isSkinValid) {
            skins[nodeId] = Skin(gltfModel, gltfModel.skins[node.skin]);
            mesh.calculateNodeBoxes(hostVertices, hostIndices, skins.at(nodeId));
        }

        for (auto& instance : instances) {
            auto& nodeData = instance.meshNodes[nodeId];
            nodeData.skeleton = GltfSkeleton(device, isSkinValid ? &skins.at(nodeId) : nullptr);
            nodeData.morphWeight = GltfMorphWeight(device, morphTargetCount);

            if (morphTargetCount > 0) {
                auto& nodeInst = instance.nodes.at(nodeId);
                if (nodeInst.weights.empty()) {
                    nodeInst.weights.assign(morphTargetCount, 0.0f);
                    nodeInst.restWeights.assign(morphTargetCount, 0.0f);
                    for (size_t wi = 0; wi < std::min(static_cast<size_t>(morphTargetCount), gltfMesh.weights.size()); wi++) {
                        nodeInst.weights[wi] = static_cast<float>(gltfMesh.weights[wi]);
                        nodeInst.restWeights[wi] = nodeInst.weights[wi];
                    }
                }
            }
        }
    }

    if (type == interfaces::Model::VertexType::pbr) {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, hostPBRVertices.size() * sizeof(interfaces::PBRVertex), hostPBRVertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cache.vertices, vertices);
    } else {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, hostVertices.size() * sizeof(interfaces::Vertex), hostVertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cache.vertices, vertices);
    }
    if (!hostIndices.empty()) {
        utils::createDeviceBuffer(device, device.device(), commandBuffer, hostIndices.size() * sizeof(uint32_t), hostIndices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, cache.indices, indices);
    }

    loadAnimations(gltfModel);

    for (auto& instance : instances) {
        updateNodes(instance.nodes, instance.meshNodes);
    }

    return true;
}

void GltfModel::createDescriptors(const utils::PhysicalDevice& device) {
    skeletonDescriptorSetLayout = interfaces::Skeleton::descriptorSetLayout(device.device());
    materialDescriptorSetLayout = interfaces::Material::descriptorSetLayout(device.device());
    morphWeightsDescriptorSetLayout = interfaces::MorphWeights::descriptorSetLayout(device.device());
    morphDeltasDescriptorSetLayout = interfaces::MorphDeltas::descriptorSetLayout(device.device());

    std::vector<const utils::vkDefault::DescriptorSetLayout*> layouts(materials.size(), &materialDescriptorSetLayout);
    for (const auto& instance : instances) {
        for (const auto& [nodeId, nodeData] : instance.meshNodes) {
            layouts.push_back(&skeletonDescriptorSetLayout);
            layouts.push_back(&morphWeightsDescriptorSetLayout);
        }
    }
    for (const auto& [_, mesh] : meshes) {
        for (const auto& primitive : mesh.primitives) {
            layouts.push_back(&morphDeltasDescriptorSetLayout);
        }
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), layouts, 1);

    for (auto& instance : instances) {
        for (auto& [nodeId, nodeData] : instance.meshNodes) {
            nodeData.skeleton.createDescriptorSet(device.device(), descriptorPool, skeletonDescriptorSetLayout);
            nodeData.morphWeight.createDescriptorSet(device.device(), descriptorPool, morphWeightsDescriptorSetLayout);
        }
    }

    for (auto& material : materials) {
        material.createDescriptorSet(device.device(), descriptorPool, materialDescriptorSetLayout);
    }

    for (auto& [_, mesh] : meshes) {
        for (auto& primitive : mesh.primitives) {
            primitive.morphDeltas.createDescriptorSet(device.device(), descriptorPool, morphDeltasDescriptorSetLayout);
        }
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

void GltfModel::render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t* primitiveCount) const {
    VkDeviceSize offsets = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertices, &offsets);
    if (VkBuffer(indices) != VK_NULL_HANDLE) {
        vkCmdBindIndexBuffer(commandBuffer, indices, 0, VK_INDEX_TYPE_UINT32);
    }

    const auto& instance = instances.at(instanceNumber);
    for (const auto& [nodeId, mesh] : meshes) {
        const auto& nodeData = instance.meshNodes.at(nodeId);
        auto descriptors = descriptorSets;
        descriptors.push_back(nodeData.skeleton.descriptorSet);
        descriptors.push_back(nodeData.morphWeight.descriptorSet);
        mesh.render(commandBuffer, pipelineLayout, descriptors, primitiveCount);
    }
}

void GltfModel::renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    const auto& instance = instances.at(instanceNumber);
    for (const auto& [nodeId, mesh] : meshes) {
        const auto& nodeData = instance.meshNodes.at(nodeId);
        auto descriptors = descriptorSets;
        descriptors.push_back(nodeData.skeleton.descriptorSet);

        const auto renderBB = (skins.find(nodeId) == skins.end() ? &GltfMesh::renderBB : &GltfMesh::renderNodeBB);
        (mesh.*renderBB)(commandBuffer, pipelineLayout, descriptors);
    }
}

std::vector<std::string> GltfModel::animationNames() const {
    return m_animationNames;
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
