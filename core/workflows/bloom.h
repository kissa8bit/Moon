#ifndef MOON_WORKFLOWS_BLOOM_H
#define MOON_WORKFLOWS_BLOOM_H

#include "workflow.h"

namespace moon::workflows {

struct BloomParameters : workflows::Parameters {
    struct{
        utils::AttachmentName bloom;
    }in;
    struct{
        utils::AttachmentName bloom;
    }out;
    uint32_t attachmentsCount{ 8 };
    float filterRadius{ 1.0f };
    float strength{ 0.8f };
    VkImageLayout inputImageLayout{ VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
};

class BloomGraphics : public Workflow
{
private:
    BloomParameters& parameters;

    std::vector<utils::Attachments> frames;
    const utils::Attachments* srcAttachment{nullptr};

    struct Downsample : public Workbody{
        const BloomParameters& parameters;
        Downsample() = default;
        Downsample(const BloomParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }downsample;

    struct Upsample : public Workbody{
        const BloomParameters& parameters;
        Upsample() = default;
        Upsample(const BloomParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }upsample;

    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    BloomGraphics() = default;
    BloomGraphics(BloomParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::workflows

#endif // MOON_WORKFLOWS_BLOOM_H
