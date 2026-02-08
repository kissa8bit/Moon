#ifndef MOON_DEFERRED_GRAPHICS_RENDER_STAGES_GRAPHICS_H
#define MOON_DEFERRED_GRAPHICS_RENDER_STAGES_GRAPHICS_H

#include <filesystem>
#include <unordered_map>

#include <workflows/workflow.h>

#include <utils/vkdefault.h>
#include <utils/texture.h>

#include <interfaces/object.h>
#include <interfaces/light.h>

#include "deferredAttachments.h"
#include "layerIndex.h"

namespace moon::deferredGraphics {

struct GraphicsParameters : workflows::Parameters {
    struct{
        utils::BufferName camera;
    }in;
    struct{
        utils::AttachmentName image;
        utils::AttachmentName bloom;
        utils::AttachmentName position;
        utils::AttachmentName normal;
        utils::AttachmentName color;
        utils::AttachmentName emission;
        utils::AttachmentName depth;
    }out;

    float minAmbientFactor{ 0.05f };
};

class Graphics : public workflows::Workflow
{
private:
    GraphicsParameters& parameters;
    DeferredAttachments deferredAttachments;
    LayerIndex layerIndex{ 0 };

    struct PipelineDesc {
        utils::vkDefault::PipelineLayout pipelineLayout;
        utils::vkDefault::Pipeline pipeline;
    };

    using PipelineDescs = std::unordered_map<interfaces::ObjectType, PipelineDesc, interfaces::ObjectType::Hasher>;

    struct Base {
        const GraphicsParameters& parameters;
        LayerIndex layerIndex{ 0 };
        const interfaces::Objects* objects{ nullptr };

        PipelineDescs                           pipelineDescs;
        utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   objectDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   skeletonDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   materialDescriptorSetLayout;
        utils::vkDefault::DescriptorPool        descriptorPool;
        utils::vkDefault::DescriptorSets        descriptorSets;

        Base(const GraphicsParameters& parameters, LayerIndex layerIndex, const interfaces::Objects* objects);

        void create(interfaces::ObjectType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
        void update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    } base;

    struct OutliningExtension {
        const Base&     parent;

        PipelineDescs   pipelineDescs;

        OutliningExtension(const Base& parent);

        void create(interfaces::ObjectType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const;
    } outlining;

    struct Lighting {
        const GraphicsParameters& parameters;
        LayerIndex layerIndex{ 0 };
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

        Lighting(const GraphicsParameters& parameters, LayerIndex layerIndex, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps);
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
             LayerIndex layerIndex,
             const interfaces::Objects* object = nullptr,
             const interfaces::Lights* lightSources = nullptr,
             const interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_RENDER_STAGES_GRAPHICS_H
