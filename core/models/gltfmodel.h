#ifndef GLTFMODEL_H
#define GLTFMODEL_H

#include <vulkan.h>
#include "tiny_gltf.h"

#include "model.h"
#include "texture.h"
#include "buffer.h"
#include "matrix.h"
#include "quaternion.h"
#include "vkdefault.h"

#include <filesystem>
#include <vector>

#define MAX_NUM_JOINTS 128u

namespace moon::models {

struct Mesh{
    struct Primitive{
        uint32_t firstIndex{0};
        uint32_t indexCount{0};
        uint32_t vertexCount{0};
        interfaces::Material* material{nullptr};
        interfaces::BoundingBox bb;
        Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, interfaces::Material* material, interfaces::BoundingBox bb);
    };

    class UniformBuffer : public utils::Buffer {
    public:
        UniformBuffer() = default;
        UniformBuffer& operator=(utils::Buffer&& other) {
            swap(other);
            return *this;
        }
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    } uniformBuffer;

    struct UniformBlock {
        math::Matrix<float,4,4> matrix;
        math::Matrix<float,4,4> jointMatrix[MAX_NUM_JOINTS]{};
        float jointcount{0};
    } uniformBlock;

    std::vector<Primitive*> primitives;

    Mesh(VkPhysicalDevice physicalDevice, VkDevice device, math::Matrix<float,4,4> matrix);
    void destroy(VkDevice device);
    ~Mesh() = default;
};

struct Node;

struct Skin {
    std::vector<math::Matrix<float,4,4>> inverseBindMatrices;
    std::vector<Node*> joints;
};

struct Node {
    VkDevice device{VK_NULL_HANDLE};
    Node* parent{nullptr};
    Mesh* mesh{nullptr};
    Skin* skin{nullptr};
    std::vector<Node*> children;

    math::Matrix<float,4,4> matrix;
    math::Vector<float,3> translation{};
    math::Vector<float,3> scale{1.0f};
    math::Quaternion<float> rotation{};

    void update();
    uint32_t meshCount() const;
    ~Node();
};

struct Animation
{
    struct AnimationChannel{
        enum PathType { TRANSLATION, ROTATION, SCALE };
        PathType path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler{
        enum InterpolationType { LINEAR, STEP, CUBICSPLINE };
        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<math::Vector<float,4>> outputsVec4;
    };

    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();
};

class GltfModel : public interfaces::Model
{
private:
    std::filesystem::path filename;
    VkDevice device{VK_NULL_HANDLE};

    utils::Buffer vertices, indices;
    utils::Buffer vertexCache, indexCache;

    utils::vkDefault::DescriptorSetLayout nodeDescriptorSetLayout;
    utils::vkDefault::DescriptorSetLayout materialDescriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;

    struct instance{
        std::unordered_map<uint32_t, Node> nodes;
        std::vector<Skin*> skins;
        std::vector<Animation> animations;
    };

    std::vector<instance> instances;
    std::vector<utils::Texture> textures;
    std::vector<interfaces::Material> materials;

    void loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer);
    void loadNode(instance* instance, VkPhysicalDevice physicalDevice, VkDevice device, Node* parent, uint32_t nodeIndex, const tinygltf::Model& model, uint32_t& indexStart);
    void loadVertexBuffer(const tinygltf::Node& node, const tinygltf::Model& model, std::vector<uint32_t>& indexBuffer, std::vector<interfaces::Vertex>& vertexBuffer);
    void loadSkins(const tinygltf::Model& gltfModel);
    void loadTextures(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer, const tinygltf::Model& gltfModel);
    void loadMaterials(const tinygltf::Model& gltfModel);
    void loadAnimations(const tinygltf::Model& gltfModel);

    Node* nodeFromIndex(uint32_t index, const std::vector<Node*>& nodes);
    void destroyCache();

    void createDescriptorPool();
    void createDescriptorSet();

public:
    GltfModel(std::filesystem::path filename, uint32_t instanceCount = 1);
    ~GltfModel();

    bool hasAnimation(uint32_t frameIndex) const override;
    float animationStart(uint32_t frameIndex, uint32_t index) const override;
    float animationEnd(uint32_t frameIndex, uint32_t index) const override;
    void updateAnimation(uint32_t frameIndex, uint32_t index, float time) override;
    void changeAnimation(uint32_t frameIndex, uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime) override;

    const VkBuffer* vertexBuffer() const override;
    const VkBuffer* indexBuffer() const override;
    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool) override;
    void render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant) override;
    void renderBB(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets) override;
};

}
#endif // GLTFMODEL_H
