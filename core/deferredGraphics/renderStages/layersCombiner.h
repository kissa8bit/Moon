#ifndef LAYERSCOMBINER_H
#define LAYERSCOMBINER_H

#include "workflow.h"

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
        std::string camera;
        std::string color;
        std::string bloom;
        std::string position;
        std::string normal;
        std::string depth;
        std::string skyboxColor;
        std::string skyboxBloom;
        std::string scattering;
        std::string sslr;
        std::string transparency;
        std::string defaultDepthTexture;
    }in;
    struct{
        std::string color;
        std::string bloom;
        std::string blur;
    }out;
    float blurDepth{ 1.0f };
    uint32_t transparentLayersCount{ 1 };
    bool enableTransparentLayers{ true };
    bool enableScatteringRefraction{ true };
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

}
#endif // LAYERSCOMBINER_H
