#ifndef MOON_WORKFLOWS_SELECTOR_H
#define MOON_WORKFLOWS_SELECTOR_H

#include "workflow.h"

#include <utils/cursor.h>

namespace moon::workflows {

struct SelectorParameters : workflows::Parameters {
    struct{
        utils::AttachmentName normal;
        utils::AttachmentName depth;
        utils::ImageName defaultDepthTexture;
    }in;
};

class SelectorGraphics : public Workflow
{
private:
    SelectorParameters& parameters;
    utils::Cursor** cursor{ nullptr };

    struct Selector : public Workbody{
        const SelectorParameters& parameters;
        Selector(const SelectorParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }selector;

    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SelectorGraphics(SelectorParameters& parameters, utils::Cursor** cursor);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::workflows

#endif // MOON_WORKFLOWS_SELECTOR_H
