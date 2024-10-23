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
    Instances instances;
    GltfMeshes meshes;
    Skins skins;

    struct Cache {
        utils::Buffer vertices;
        utils::Buffer indices;
    } cache;

    using BoxMap = std::unordered_map<uint32_t, math::box>;
    BoxMap boxMap;

    bool loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadVertices(const tinygltf::Model& gltfModel, const tinygltf::Node& node, interfaces::Indices& indices, interfaces::Vertices& vertices);
    void loadTextures(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadAnimations(const tinygltf::Model& gltfModel);
    void loadMaterials(const tinygltf::Model& gltfModel);

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
