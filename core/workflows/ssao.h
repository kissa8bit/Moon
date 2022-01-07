#ifndef SSAO_H
#define SSAO_H

#include "workflow.h"

namespace moon::workflows {

struct SSAOParameters : workflows::Parameters{
    struct{
        std::string camera;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string ssao;
    }out;
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

}
#endif // SSAO_H
