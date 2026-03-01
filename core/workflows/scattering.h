#ifndef MOON_WORKFLOWS_SCATTERING_H
#define MOON_WORKFLOWS_SCATTERING_H

#include "workflow.h"

#include <interfaces/light.h>

namespace moon::workflows {

struct ScatteringParameters : workflows::Parameters{
    struct{
        utils::BufferName camera;
        utils::AttachmentName depth;
    }in;
    struct{
        utils::AttachmentName scattering;
    }out;
    float   density { 0.0f };
    int32_t steps   { 50 };
};

class Scattering : public Workflow
{
private:
    ScatteringParameters& parameters;
    utils::Attachments frame;

    struct Lighting : Workbody {
        const ScatteringParameters& parameters;
        const interfaces::Lights* lightSources{ nullptr };
        const interfaces::DepthMaps* depthMaps{ nullptr };

        struct PipelineDesc {
            utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
            utils::vkDefault::PipelineLayout pipelineLayout;
            utils::vkDefault::Pipeline pipeline;
        };
        using PipelineDescs = std::unordered_map<interfaces::LightType, PipelineDesc, interfaces::LightType::Hasher>;

        PipelineDescs                             pipelineDescs;
        utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;

        Lighting(const ScatteringParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps)
            : parameters(parameters), lightSources(lightSources), depthMaps(depthMaps)
        {}

        void createPipeline(interfaces::LightType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass pRenderPass);
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override {}
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }lighting;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    Scattering(ScatteringParameters& parameters, const interfaces::Lights* lightSources = nullptr, const interfaces::DepthMaps* depthMaps = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::workflows

#endif // MOON_WORKFLOWS_SCATTERING_H
