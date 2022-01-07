#ifndef BLOOM_H
#define BLOOM_H

#include "workflow.h"

namespace moon::workflows {

struct BloomParameters : workflows::Parameters {
    struct{
        std::string bloom;
    }in;
    struct{
        std::string bloom;
    }out;
    uint32_t blitAttachmentsCount{ 0 };
    float blitFactor{ 1.5f };
    float xSamplerStep{ 1.5f };
    float ySamplerStep{ 1.5f };
    VkImageLayout inputImageLayout{ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
};

class BloomGraphics : public Workflow
{
private:
    BloomParameters& parameters;

    std::vector<utils::Attachments> frames;
    utils::Attachments bufferAttachment;
    const utils::Attachments* srcAttachment{nullptr};

    struct Filter : public Workbody{
        const BloomParameters& parameters;
        Filter() = default;
        Filter(const BloomParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }filter;

    struct Bloom : public Workbody{
        const BloomParameters& parameters;
        Bloom() = default;
        Bloom(const BloomParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }bloom;

    void render(VkCommandBuffer commandBuffer, const utils::Attachments& image, uint32_t frameNumber, uint32_t framebufferIndex, Workbody* worker);

    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    BloomGraphics() = default;
    BloomGraphics(BloomParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // BLOOM_H
