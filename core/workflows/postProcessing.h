#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "workflow.h"

namespace moon::workflows {

struct PostProcessingParameters : workflows::Parameters{
    struct{
        std::string baseColor;
        std::string blur;
        std::string bloom;
        std::string ssao;
        std::string boundingBox;
    }in;
    struct{
        std::string postProcessing;
    }out;
};

class PostProcessingGraphics : public Workflow
{
private:
    PostProcessingParameters& parameters;
    utils::Attachments frame;

    struct PostProcessing : public Workbody{
        const PostProcessingParameters& parameters;
        PostProcessing(const PostProcessingParameters& parameters) : parameters(parameters) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    } postProcessing;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    PostProcessingGraphics(PostProcessingParameters& parameters);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // POSTPROCESSING_H
