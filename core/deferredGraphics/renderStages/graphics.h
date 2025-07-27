#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "vkdefault.h"
#include "workflow.h"
#include "deferredAttachments.h"

#include "object.h"
#include "light.h"
#include "texture.h"

#include <filesystem>
#include <unordered_map>

namespace moon::deferredGraphics {

struct GraphicsParameters : workflows::Parameters {
    struct{
        std::string camera;
    }in;
    struct{
        std::string image;
        std::string blur;
        std::string bloom;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string transparency;
    }out;

    bool        transparencyPass{ false };
    bool        enableTransparency{ false };
    uint32_t    transparencyNumber{ 0 };
    float       minAmbientFactor{ 0.05f };
};

class Graphics : public workflows::Workflow
{
private:
    GraphicsParameters& parameters;
    DeferredAttachments deferredAttachments;

    struct Base {
        const GraphicsParameters& parameters;
        const interfaces::Objects* objects{ nullptr };

        struct PipelineDesc {
            utils::vkDefault::PipelineLayout pipelineLayout;
            utils::vkDefault::Pipeline pipeline;
        };
        using PipelineDescs = std::unordered_map<interfaces::ObjectType, PipelineDesc, interfaces::ObjectType::Hasher>;

        PipelineDescs                           pipelineDescs;
        utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   objectDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   skeletonDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   materialDescriptorSetLayout;
        utils::vkDefault::DescriptorPool        descriptorPool;
        utils::vkDefault::DescriptorSets        descriptorSets;

        Base(const GraphicsParameters& parameters, const interfaces::Objects* objects);

        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
        void update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    } base;

    struct OutliningExtension {
        const Base& parent;

        utils::vkDefault::PipelineLayout  pipelineLayout;
        utils::vkDefault::Pipeline        pipeline;

        OutliningExtension(const Base& parent);

        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    } outlining;

    struct Lighting {
        const GraphicsParameters& parameters;
        const interfaces::Lights* lightSources{ nullptr };
        const interfaces::DepthMaps* depthMaps{ nullptr };

        struct PipelineDesc {
            utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
            utils::vkDefault::PipelineLayout pipelineLayout;
            utils::vkDefault::Pipeline pipeline;
        };
        using PipelineDescs = std::unordered_map<interfaces::LightType, PipelineDesc, interfaces::LightType::Hasher>;

        PipelineDescs                               pipelineDescs;
        utils::vkDefault::DescriptorSetLayout       descriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout       shadowDescriptorSetLayout;
        utils::vkDefault::DescriptorPool            descriptorPool;
        utils::vkDefault::DescriptorSets            descriptorSets;

        Lighting(const GraphicsParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps);
        void createPipeline(interfaces::LightType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);

        void create(VkDevice device, VkRenderPass renderPass);
        void update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const;
    } lighting;

    struct AmbientLighting {
        const Lighting& parent;

        utils::vkDefault::PipelineLayout  pipelineLayout;
        utils::vkDefault::Pipeline        pipeline;

        AmbientLighting(const Lighting& parent);

        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    } ambientLighting;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void createPipelines();

    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    Graphics(GraphicsParameters& parameters,
             const interfaces::Objects* object = nullptr,
             const interfaces::Lights* lightSources = nullptr,
             const interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // GRAPHICS_H
