#ifndef MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERSCOMBINER_H
#define MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERSCOMBINER_H

#include <workflows/workflow.h>

#include "layerIndex.h"

namespace moon::deferredGraphics {

struct LayersCombinerAttachments{
    utils::Attachments color;
    utils::Attachments bloom;
    utils::Attachments blur;

    static inline uint32_t size() { return 3;}
    inline utils::Attachments* operator&(){ return &color;}
};

struct LayersCombinerParameters : workflows::Parameters {
    struct{
        utils::BufferName camera;
        utils::AttachmentName color;
        utils::AttachmentName bloom;
        utils::AttachmentName position;
        utils::AttachmentName normal;
        utils::AttachmentName depth;
        utils::AttachmentName skyboxColor;
        utils::AttachmentName skyboxBloom;
        utils::AttachmentName scattering;
        utils::AttachmentName sslr;
        utils::ImageName defaultDepthTexture;
    }in;
    struct{
        utils::AttachmentName color;
        utils::AttachmentName bloom;
        utils::AttachmentName blur;
    }out;

    LayerIndex layersCount{ 1 };
};

class LayersCombiner : public workflows::Workflow
{
private:
    LayersCombinerParameters& parameters;
    LayersCombinerAttachments frame;

    struct Combiner : public workflows::Workbody {
        const LayersCombinerParameters& parameters;
        Combiner(const LayersCombinerParameters& parameters) : parameters(parameters){};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    } combiner;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();

    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    LayersCombiner(LayersCombinerParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERSCOMBINER_H
