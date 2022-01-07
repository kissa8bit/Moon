#ifndef SKYBOX_H
#define SKYBOX_H

#include "workflow.h"
#include "vkdefault.h"
#include "object.h"

namespace moon::workflows {

struct SkyboxAttachments
{
    utils::Attachments color;
    utils::Attachments bloom;

    inline uint32_t size() const{
        return 2;
    }
    inline utils::Attachments* operator&(){
        return &color;
    }
};

struct SkyboxParameters : workflows::Parameters {
    struct{
        std::string camera;
    }in;
    struct{
        std::string baseColor;
        std::string bloom;
    }out;
};

class SkyboxGraphics : public Workflow
{
private:
    SkyboxParameters& parameters;
    SkyboxAttachments frame;

    struct Skybox : public Workbody{
        const SkyboxParameters& parameters;
        utils::vkDefault::DescriptorSetLayout objectDescriptorSetLayout;
        const interfaces::Objects* objects{nullptr};

        Skybox(const SkyboxParameters& parameters, const interfaces::Objects* objects) : parameters(parameters), objects(objects) {};
        void create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) override;
    }skybox;

    void createAttachments(utils::AttachmentsDatabase& aDatabase);
    void createRenderPass();
    void createFramebuffers();
    void updateCommandBuffer(uint32_t frameNumber) override;

public:
    SkyboxGraphics(SkyboxParameters& parameters, const interfaces::Objects* object = nullptr);

    void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) override;
    void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) override;
};

}
#endif // SKYBOX_H
