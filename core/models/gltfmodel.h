#ifndef GLTFMODEL_H
#define GLTFMODEL_H

#include <filesystem>
#include <vector>

#include <vulkan.h>

#include "model.h"
#include "buffer.h"
#include "vkdefault.h"

#include "linearAlgebra.h"

#include "gltfmodel/gltfmesh.h"
#include "gltfmodel/instance.h"
#include "gltfmodel/tinyGLTF.h"

namespace moon::models {

class GltfModel : public interfaces::Model {
private:
    std::filesystem::path filename;
    utils::Buffer vertexCache, indexCache;
    Instances instances;
    GltfMeshes meshes;
    Skins skins;

    using BoxMap = std::unordered_map<uint32_t, math::box>;
    BoxMap boxMap;

    bool loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadVertexBuffer(const tinygltf::Model& gltfModel, const tinygltf::Node& node, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer);
    void loadSkins(const tinygltf::Model& gltfModel);
    void loadTextures(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadMaterials(const tinygltf::Model& gltfModel);
    void loadAnimations(const tinygltf::Model& gltfModel);

    void createDescriptors(const utils::PhysicalDevice& device);
    void destroyCache();

public:
    GltfModel(std::filesystem::path filename, uint32_t instanceCount = 1);

    std::vector<interfaces::Animation*> animations(uint32_t instanceNumber) override;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const override;
    void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const override;
};

}
#endif // GLTFMODEL_H
