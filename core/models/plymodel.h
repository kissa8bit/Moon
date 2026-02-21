#ifndef MOON_MODELS_PLYMODEL_H
#define MOON_MODELS_PLYMODEL_H

#include <filesystem>
#include <vector>

#include <interfaces/model.h>

#include <utils/buffer.h>
#include <utils/vkdefault.h>

#include <math/linearAlgebra.h>

namespace moon::models {

class PlyModel : public interfaces::Model {
private:
    std::filesystem::path filename;
    interfaces::Mesh mesh;
    interfaces::Skeleton skeleton;

    struct Cache {
        utils::Buffer vertices;
        utils::Buffer indices;
    } cache;

    bool loadFromFile(const utils::PhysicalDevice& physicalDevice, VkCommandBuffer commandBuffer);
    void createDescriptors(VkDevice device);
    void destroyCache();

public:
    PlyModel(const std::filesystem::path& filename,
             const math::vec4& baseColorFactor = math::vec4(1.0f),
             const math::vec4& diffuseFactor = math::vec4(1.0f),
             const math::vec4& specularFactor = math::vec4(1.0f),
             const float metallicFactor = 0.0f,
             const float roughnessFactor = 0.0f,
             const interfaces::Material::PbrWorkflow workflow = interfaces::Material::PbrWorkflow::METALLIC_ROUGHNESS);

    interfaces::Material& material();
    const interfaces::Material& material() const;
    math::box boundingBox() const;

    std::vector<interfaces::Animation*> animations(uint32_t instanceNumber) override;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const override;
    void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const override;
};

} // moon::models

#endif // MOON_MODELS_PLYMODEL_H
