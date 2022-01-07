#ifndef MOON_WORKFLOWS_SHADOW_H
#define MOON_WORKFLOWS_SHADOW_H

#include "workflow.h"

#include <interfaces/object.h>
#include <interfaces/light.h>

namespace moon::workflows {

struct ShadowGraphicsParameters : workflows::Parameters {};

class ShadowGraphics : public Workflow
{
private:
    ShadowGraphicsParameters& parameters;
    using FramebuffersMap = std::unordered_map<const utils::DepthMap*, utils::vkDefault::Framebuffers>;
    FramebuffersMap framebuffersMap;

    struct PipelineDesc {
        utils::vkDefault::PipelineLayout pipelineLayout;
        utils::vkDefault::Pipeline pipeline;
    };

    using PipelineDescs = std::unordered_map<interfaces::ObjectType, PipelineDesc, interfaces::ObjectType::Hasher>;

    struct Shadow : public Workbody{
        const ShadowGraphicsParameters& parameters;
        const interfaces::Objects* objects{ nullptr };
        interfaces::DepthMaps* depthMaps{ nullptr };

        PipelineDescs                           pipelineDescs;
        utils::vkDefault::DescriptorSetLayout   lightDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   objectDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   skeletonDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout   materialDescriptorSetLayout;

        Shadow(const ShadowGraphicsParameters& parameters, const interfaces::Objects* objects, interfaces::DepthMaps* depthMaps)
            : parameters(parameters), objects(objects), depthMaps(depthMaps) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override {};
        void create(interfaces::ObjectType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass);
    }shadow;

    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, interfaces::Light* lightSource, const utils::DepthMap& depthMap, const utils::vkDefault::Framebuffer& framebuffer);
    void createRenderPass();
    void updateCommandBuffer(uint32_t frameNumber) override;
    void updateFramebuffersMap();

public:
    ShadowGraphics(ShadowGraphicsParameters& parameters, const interfaces::Objects* objects = nullptr, interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase&) override;
    void updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase&) override{};
};

} // moon::workflows

#endif // MOON_WORKFLOWS_SHADOW_H
