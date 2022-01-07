#ifndef MOON_INTERFACES_MODEL_H
#define MOON_INTERFACES_MODEL_H

#include <vector>
#include <limits>

#include <vulkan.h>

#include <math/linearAlgebra.h>

#include <utils/vkdefault.h>
#include <utils/texture.h>
#include <utils/device.h>

namespace moon::interfaces {

struct Material {
    struct Buffer {
        math::vec4  baseColorFactor{ 0.0f };
        math::vec4  emissiveFactor{ 0.0f };
        math::vec4  diffuseFactor{ 0.0f };
        math::vec4  specularFactor{ 0.0f };
        float       workflow{ 0.0f };
        int         colorTextureSet{ -1 };
        int         physicalDescriptorTextureSet{ -1 };
        int         normalTextureSet{ -1 };
        int         occlusionTextureSet{ -1 };
        int         emissiveTextureSet{ -1 };
        float       metallicFactor{ 0.0f };
        float       roughnessFactor{ 0.0f };
        float       alphaMask{ 0.0f };
        float       alphaMaskCutoff{ 0.0f };
        uint32_t    primitive{ 0 };

        Buffer(const Material& material, uint32_t primitive);
    };

    enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
    enum PbrWorkflow { METALLIC_ROUGHNESS, SPECULAR_GLOSSINESS };

    static constexpr uint32_t metallicIndex = 0;
    static constexpr uint32_t roughnessIndex = 1;

    struct TextureParameters {
        const utils::Texture* texture{ nullptr };
        math::vec4 factor{ 1.0f };
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

    Material() = default;
    Material(const utils::Texture* empty);
    void createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout);
    Buffer buffer(uint32_t primitive) const;

    static utils::vkDefault::DescriptorSetLayout descriptorSetLayout(VkDevice device);
};

using Materials = std::vector<Material>;

struct Range {
    uint32_t first{ 0 };
    uint32_t count{ 0 };

    uint32_t last() const {return first + count;}
    void setLast(uint32_t last) { count = last - first; }
};

struct Primitive {
    struct { Range range{}; } index;
    struct { Range range{}; } vertex;
    const interfaces::Material* material{ nullptr };
    math::box bb;

    Primitive() = default;
    Primitive(const Range& indexRange, const Range vertexRange, const interfaces::Material* material, math::box bb)
        : index({indexRange}), vertex({vertexRange}), material(material), bb(bb)
    {}
};

struct Skeleton {
    utils::Buffer deviceBuffer;
    struct Buffer {
        static constexpr auto maxJoints = 256u;
        alignas(16) math::mat4 matrix{math::mat4::identity()};
        alignas(16) math::mat4 jointMatrix[maxJoints]{};
    } hostBuffer;
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    void createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout);

    static utils::vkDefault::DescriptorSetLayout descriptorSetLayout(VkDevice device);
};

struct Mesh {
    std::vector<Primitive> primitives;

    void render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const;
    void renderBB(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const;
};

using Indices = std::vector<uint32_t>;

struct SimpleVertex {
    alignas(16) math::vec3 pos{ 0.0f };
    alignas(16) math::vec3 normal{ 0.0f };
    alignas(8)  math::vec2 uv0{ 0.0f };

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

using SimpleVertices = std::vector<SimpleVertex>;

struct PBRVertex {
    alignas(16) math::vec3 pos{ 0.0f };
    alignas(16) math::vec3 normal{ 0.0f };
    alignas(8)  math::vec2 uv0{ 0.0f };
    alignas(8)  math::vec2 uv1{ 0.0f };
    alignas(16) math::vec3 tangent{ 0.0f };

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

using PBRVertices = std::vector<PBRVertex>;

struct Vertex {
    alignas(16) math::vec3 pos{ 0.0f };
    alignas(16) math::vec3 normal{ 0.0f };
    alignas(8)  math::vec2 uv0{ 0.0f };
    alignas(8)  math::vec2 uv1{ 0.0f };
    alignas(16) math::vec4 joint{ -1.0f };
    alignas(16) math::vec4 weight{ 1.0f, 0.0f, 0.0f, 0.0f };
    alignas(16) math::vec3 tangent{ 0.0f };

    static VkVertexInputBindingDescription getBindingDescription();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

using Vertices = std::vector<Vertex>;

class Animation {
public:
    virtual ~Animation() {};
    virtual void setChangeTime(float changetime) = 0;
    virtual bool update(float time) = 0;
    virtual float duration() const = 0;
};

class Model {
public:
    enum VertexType : uint32_t
    {
        non = 0ul,
        baseSimple = 1ul << 0,
        basePBR = 1ul << 1,
        base = 1ul << 2,
    };

    virtual ~Model() = default;

    virtual std::vector<Animation*> animations(uint32_t instanceNumber) = 0;

    virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) = 0;
    virtual void render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const = 0;
    virtual void renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const = 0;

    VertexType vertexType() const { return type; }

protected:
    VertexType type{};

    utils::vkDefault::DescriptorSetLayout skeletonDescriptorSetLayout;
    utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;

    utils::Buffer vertices, indices;
    utils::Textures textures;
    interfaces::Materials materials;
};

} // moon::interfaces

#endif // MOON_INTERFACES_MODEL_H
