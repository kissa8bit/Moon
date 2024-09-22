#ifndef PLYMODEL_H
#define PLYMODEL_H

#include "model.h"
#include "buffer.h"
#include "matrix.h"
#include "vkdefault.h"

#include <filesystem>
#include <vector>

namespace moon::models {

class PlyModel : public interfaces::Model{
private:
    std::filesystem::path filename;
    utils::Buffer vertexCache, indexCache;
    interfaces::Mesh mesh;

    void loadFromFile(const utils::PhysicalDevice& physicalDevice, VkCommandBuffer commandBuffer);
    void createDescriptors(VkDevice device);
    void destroyCache();

public:
    PlyModel(const std::filesystem::path& filename,
             const math::Vector<float, 4>& baseColorFactor = math::Vector<float, 4>(1.0f),
             const math::Vector<float, 4>& diffuseFactor = math::Vector<float, 4>(1.0f),
             const math::Vector<float, 4>& specularFactor = math::Vector<float, 4>(1.0f),
             const float metallicFactor = 0.0f,
             const float roughnessFactor = 0.0f,
             const interfaces::Material::PbrWorkflow workflow = interfaces::Material::PbrWorkflow::METALLIC_ROUGHNESS);

    interfaces::Material& material();
    const interfaces::Material& material() const;
    interfaces::BoundingBox boundingBox() const;

    bool hasAnimation(uint32_t) const override {return false;}
    float animationStart(uint32_t, uint32_t) const override {return 0.0f;}
    float animationEnd(uint32_t, uint32_t) const override {return 0.0f;}
    void updateAnimation(uint32_t, uint32_t, float) override {};
    void changeAnimation(uint32_t, uint32_t, uint32_t, float, float, float) override {};

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const override;
    void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const override;
};

}
#endif // PLYMODEL_H
