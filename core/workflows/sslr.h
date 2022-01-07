#ifndef SSLR_H
#define SSLR_H

#include "workflow.h"

namespace moon::workflows {

struct SSLRParameters : workflows::Parameters{
    struct{
        std::string camera;
        std::string position;
        std::string normal;
        std::string color;
        std::string depth;
        std::string firstTransparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string sslr;
    }out;
};

class SSLRGraphics : public Workflow
{
private:
    SSLRParameters& parameters;
    utils::Attachments frame;

    struct SSLR : public Workbody{
        const SSLRParameters& parameters;
        SSLR(const SSLRParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    } sslr;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SSLRGraphics(SSLRParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // SSLR_H
