#ifndef GLTFMODEL_H
#define GLTFMODEL_H

#include <filesystem>
#include <vector>

#include <vulkan.h>

#include "model.h"
#include "buffer.h"
#include "matrix.h"
#include "quaternion.h"
#include "vkdefault.h"

#include "gltfmodel/instance.h"
#include "gltfmodel/tinyGLTF.h"

namespace moon::models {

class GltfModel : public interfaces::Model {
private:
    std::filesystem::path filename;
    utils::Buffer vertexCache, indexCache;
    Instances instances;

    void loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadNode(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, NodeMap& instance, Node* parent, uint32_t nodeIndex, uint32_t& indexStart);
    void loadVertexBuffer(const tinygltf::Model& gltfModel, const tinygltf::Node& node, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer);
    void loadSkins(const tinygltf::Model& gltfModel);
    void loadTextures(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadMaterials(const tinygltf::Model& gltfModel);
    void loadAnimations(const tinygltf::Model& gltfModel);

    void createDescriptors(VkDevice device);
    void destroyCache();

public:
    GltfModel(std::filesystem::path filename, uint32_t instanceCount = 1);

    bool hasAnimation(uint32_t instanceIndex) const override;
    float animationStart(uint32_t instanceIndex, uint32_t index) const override;
    float animationEnd(uint32_t instanceIndex, uint32_t index) const override;
    void updateAnimation(uint32_t instanceIndex, uint32_t index, float time) override;
    void changeAnimation(uint32_t instanceIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime) override;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const override;
    void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const override;
};

}
#endif // GLTFMODEL_H
