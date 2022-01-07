#ifndef BLUR_H
#define BLUR_H

#include "workflow.h"

namespace moon::workflows {

struct GaussianBlurParameters : workflows::Parameters {
    struct{
        std::string blur;
    }in;
    struct{
        std::string blur;
    }out;
    float blurDepth{ 1.0f };
};

class GaussianBlur : public Workflow
{
private:
    GaussianBlurParameters& parameters;
    utils::Attachments bufferAttachment;
    utils::Attachments frame;

    struct Blur : public Workbody{
        uint32_t subpassNumber{ 0 };
        const GaussianBlurParameters& parameters;
        Blur(const GaussianBlurParameters& parameters, uint32_t subpassNumber) : parameters(parameters), subpassNumber(subpassNumber) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    };
    Blur xblur;
    Blur yblur;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    GaussianBlur(GaussianBlurParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // BLUR_H
