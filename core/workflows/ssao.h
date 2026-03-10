#ifndef MOON_WORKFLOWS_SSAO_H
#define MOON_WORKFLOWS_SSAO_H

#include "workflow.h"

namespace moon::workflows {

struct SSAOParameters : workflows::Parameters{
    struct{
        utils::BufferName camera;
        utils::AttachmentName normal;
        utils::AttachmentName color;
        utils::AttachmentName depth;
        utils::ImageName defaultDepthTexture;
    }in;
    struct{
        utils::AttachmentName ssao;
    }out;

    int32_t kernelSize { 32 };
    float   radius     { 0.5f };
    float   aoFactor   { 1.0f };
    float   aoPower    { 4.0f };
};

class SSAOGraphics : public Workflow
{
private:
    SSAOParameters& parameters;
    utils::Attachments frame;

    struct SSAO : public Workbody{
        const SSAOParameters& parameters;
        SSAO(const SSAOParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    } ssao;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SSAOGraphics(SSAOParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::workflows

#endif // MOON_WORKFLOWS_SSAO_H
