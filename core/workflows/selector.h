#ifndef SELECTOR_H
#define SELECTOR_H

#include "workflow.h"
#include "cursor.h"

namespace moon::workflows {

struct SelectorParameters : workflows::Parameters {
    struct{
        std::string storageBuffer;
        std::string position;
        std::string depth;
        std::string transparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string selector;
    }out;
    uint32_t transparentLayersCount{ 1 };
};

class SelectorGraphics : public Workflow
{
private:
    SelectorParameters& parameters;
    utils::Attachments frame;
    utils::Cursor** cursor{ nullptr };

    struct Selector : public Workbody{
        const SelectorParameters& parameters;
        Selector(const SelectorParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }selector;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SelectorGraphics(SelectorParameters& parameters, utils::Cursor** cursor);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // SELECTOR_H
