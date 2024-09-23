#ifndef MODEL_H
#define MODEL_H

#include <vulkan.h>
#include <vector>
#include <limits>

#include "vector.h"
#include "matrix.h"
#include "vkdefault.h"
#include "texture.h"
#include "device.h"

namespace moon::interfaces {

struct BoundingBox{
    alignas(16) math::Vector<float,3> min{std::numeric_limits<float>::max()};
    alignas(16) math::Vector<float,3> max{0.0f};

    BoundingBox() = default;
    BoundingBox(math::Vector<float,3> min, math::Vector<float,3> max);
};

struct Material {
    enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
    enum PbrWorkflow { METALLIC_ROUGHNESS, SPECULAR_GLOSSINESS };

    static constexpr uint32_t metallicIndex = 0;
    static constexpr uint32_t roughnessIndex = 1;

    struct TextureParameters {
        const utils::Texture* texture{ nullptr };
        math::Vector<float, 4> factor{ 1.0f };
        int8_t coordSet{ -1 };

        TextureParameters() = default;
        TextureParameters(const utils::Texture* texture) : texture(texture) {}
    };

    AlphaMode alphaMode = ALPHAMODE_OPAQUE;
    PbrWorkflow pbrWorkflows = METALLIC_ROUGHNESS;

    TextureParameters baseColor;
    TextureParameters metallicRoughness;
    TextureParameters normal;
    TextureParameters occlusion;
    TextureParameters emissive;
    struct {
        TextureParameters specularGlossiness;
        TextureParameters diffuse;
    } extensions;

    float alphaCutoff{1.0f};

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    void createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout);
};

using Materials = std::vector<Material>;

struct MaterialBlock {
    math::Vector<float,4> baseColorFactor{0.0f};
    math::Vector<float,4> emissiveFactor{0.0f};
    math::Vector<float,4> diffuseFactor{0.0f};
    math::Vector<float,4> specularFactor{0.0f};
    float       workflow{0.0f};
    int         colorTextureSet{-1};
    int         physicalDescriptorTextureSet{-1};
    int         normalTextureSet{-1};
    int         occlusionTextureSet{-1};
    int         emissiveTextureSet{-1};
    float       metallicFactor{0.0f};
    float       roughnessFactor{0.0f};
    float       alphaMask{0.0f};
    float       alphaMaskCutoff{0.0f};
    uint32_t    primitive;

    MaterialBlock(const Material& material, uint32_t primitive);
};

struct MeshBlock {
    static constexpr auto maxJoints = 128u;
    float jointcount{ 0 };
    alignas(16) math::Matrix<float, 4, 4> matrix{ 1.0f };
    alignas(16) math::Matrix<float, 4, 4> jointMatrix[maxJoints]{};

    size_t size() const {
        return 4 * sizeof(float) + (jointcount + 1) * sizeof(math::Matrix<float, 4, 4>);
    }
};

struct Primitive {
    uint32_t firstIndex{ 0 };
    uint32_t indexCount{ 0 };
    uint32_t vertexCount{ 0 };
    const interfaces::Material* material{ nullptr };
    interfaces::BoundingBox bb;

    Primitive() = default;
    Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, const interfaces::Material* material, interfaces::BoundingBox bb)
        : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material), bb(bb)
    {}
};

struct Mesh {
    utils::Buffer uniformBuffer;
    interfaces::MeshBlock uniformBlock;

    std::vector<Primitive> primitives;
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    bool empty() const;
    void createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout);
    void render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const;
    void renderBB(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const;
};

struct Vertex {
    alignas(16) math::Vector<float, 3> pos{ 0.0f };
    alignas(16) math::Vector<float, 3> normal{ 0.0f };
    alignas(8)  math::Vector<float, 2> uv0{ 0.0f };
    alignas(8)  math::Vector<float, 2> uv1{ 0.0f };
    alignas(16) math::Vector<float, 4> joint0{ 0.0f };
    alignas(16) math::Vector<float, 4> weight0{ 1.0f, 0.0f, 0.0f, 0.0f };
    alignas(16) math::Vector<float, 3> tangent{ 0.0f };
    alignas(16) math::Vector<float, 3> bitangent{ 0.0f };

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

class Model {
protected:
    utils::vkDefault::DescriptorSetLayout meshDescriptorSetLayout;
    utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;

    utils::Buffer vertices, indices;
    utils::Textures textures;
    interfaces::Materials materials;

public:
    virtual ~Model(){};

    virtual bool hasAnimation(uint32_t instanceIndex) const = 0;
    virtual float animationStart(uint32_t instanceIndex, uint32_t index) const = 0;
    virtual float animationEnd(uint32_t instanceIndex, uint32_t index) const = 0;
    virtual void updateAnimation(uint32_t instanceIndex, uint32_t index, float time) = 0;
    virtual void changeAnimation(uint32_t instanceIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime) = 0;

    virtual const VkBuffer* vertexBuffer() const;
    virtual const VkBuffer* indexBuffer() const;
    virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) = 0;
    virtual void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const = 0;
    virtual void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const = 0;

    static utils::vkDefault::DescriptorSetLayout createMeshDescriptorSetLayout(VkDevice device);
    static utils::vkDefault::DescriptorSetLayout createMaterialDescriptorSetLayout(VkDevice device);
};

}
#endif // MODEL_H
