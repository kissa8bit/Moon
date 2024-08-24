#ifndef PLYMODEL_H
#define PLYMODEL_H

#include "model.h"
#include "texture.h"
#include "buffer.h"
#include "matrix.h"
#include "vkdefault.h"

#include <filesystem>

#define MAX_NUM_JOINTS 128u

namespace moon::models {

class PlyModel : public moon::interfaces::Model{
private:
    std::filesystem::path filename;
    bool created{false};
    moon::utils::Texture emptyTexture;
    VkDevice device{VK_NULL_HANDLE};

    moon::utils::Buffer vertices, indices;
    moon::utils::Buffer vertexCache, indexCache;

    uint32_t indexCount{0};

    moon::utils::vkDefault::DescriptorSetLayout nodeDescriptorSetLayout;
    moon::utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;
    moon::utils::vkDefault::DescriptorPool descriptorPool;

    moon::interfaces::Material material;
    moon::interfaces::MaterialBlock materialBlock{};

    class UniformBuffer : public moon::utils::Buffer {
    public:
        UniformBuffer() = default;
        UniformBuffer& operator=(utils::Buffer && other) {
            swap(other);
            return *this;
        }
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    } uniformBuffer;

    struct UniformBlock {
        moon::math::Matrix<float,4,4> mat;
        moon::math::Matrix<float,4,4> jointMatrix[MAX_NUM_JOINTS]{};
        float jointcount{0};
    } uniformBlock;

    moon::interfaces::BoundingBox bb;

    moon::math::Vector<float,3> maxSize{0.0f};

    void loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer);
    void destroyCache();

    void createDescriptorPool();
    void createDescriptorSet();

public:
    PlyModel(std::filesystem::path filename,
             moon::math::Vector<float, 4> baseColorFactor = moon::math::Vector<float, 4>(1.0f,1.0f,1.0f,1.0f),
             moon::math::Vector<float, 4> diffuseFactor = moon::math::Vector<float, 4>(1.0f),
             moon::math::Vector<float, 4> specularFactor = moon::math::Vector<float, 4>(1.0f),
             float metallicFactor = 0.0f,
             float roughnessFactor = 0.5f,
             float workflow = 1.0f);

    moon::interfaces::MaterialBlock& getMaterialBlock();
    const moon::math::Vector<float,3> getMaxSize() const;

    bool hasAnimation(uint32_t) const override {return false;}
    float animationStart(uint32_t, uint32_t) const override {return 0.0f;}
    float animationEnd(uint32_t, uint32_t) const override {return 0.0f;}
    void updateAnimation(uint32_t, uint32_t, float) override {};
    void changeAnimation(uint32_t, uint32_t, uint32_t, float, float, float) override {};

    const VkBuffer* vertexBuffer() const override;
    const VkBuffer* indexBuffer() const override;
    void create(const moon::utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) override;
    void renderBB(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets) override;
};

}
#endif // PLYMODEL_H
