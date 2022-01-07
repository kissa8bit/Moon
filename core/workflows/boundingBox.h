#ifndef MOON_WORKFLOWS_BOUNDINGBOX_H
#define MOON_WORKFLOWS_BOUNDINGBOX_H

#include "workflow.h"

#include <interfaces/object.h>

namespace moon::workflows {

struct BoundingBoxParameters : workflows::Parameters {
    struct{
        std::string camera;
    }in;
    struct{
        std::string boundingBox;
    }out;
};

class BoundingBoxGraphics : public Workflow
{
private:
    BoundingBoxParameters& parameters;
    utils::Attachments frame;

    struct BoundingBox : public Workbody{
        const BoundingBoxParameters& parameters;
        utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        utils::vkDefault::DescriptorSetLayout skeletonDescriptorSetLayout;
        const interfaces::Objects* objects;

        BoundingBox(const BoundingBoxParameters& parameters, const interfaces::Objects* objects) : parameters(parameters), objects(objects) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
        void render(uint32_t frameNumber, VkCommandBuffer commandBuffers);
    }box;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    BoundingBoxGraphics(BoundingBoxParameters& parameters, const interfaces::Objects* objects = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

} // moon::workflows

#endif // MOON_WORKFLOWS_BOUNDINGBOX_H
