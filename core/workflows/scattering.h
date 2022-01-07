#ifndef SCATTERING_H
#define SCATTERING_H

#include "workflow.h"
#include "vkdefault.h"
#include "light.h"

namespace moon::workflows {

struct ScatteringParameters : workflows::Parameters{
    struct{
        std::string camera;
        std::string depth;
    }in;
    struct{
        std::string scattering;
    }out;
};

class Scattering : public Workflow
{
private:
    ScatteringParameters& parameters;
    utils::Attachments frame;

    struct Lighting : Workbody{
        const ScatteringParameters& parameters;

        utils::vkDefault::DescriptorSetLayout     shadowDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayoutMap  lightDescriptorSetLayoutMap;
        utils::vkDefault::PipelineLayoutMap       pipelineLayoutMap;
        utils::vkDefault::PipelineMap             pipelinesMap;

        const interfaces::Lights* lightSources{ nullptr };
        const interfaces::DepthMaps* depthMaps{ nullptr };

        Lighting(const ScatteringParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps)
            : parameters(parameters), lightSources(lightSources), depthMaps(depthMaps)
        {}

        void createPipeline(uint8_t mask, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass pRenderPass);
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
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

}
#endif // SCATTERING_H
