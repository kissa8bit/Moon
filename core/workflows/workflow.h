#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <vulkan.h>
#include "attachments.h"
#include "buffer.h"
#include "vkdefault.h"

#include <filesystem>
#include <unordered_map>
#include <memory>

namespace moon::workflows {

enum class ShaderType
{
    Vertex,
    Fragment
};

using ShaderNames = std::unordered_map<ShaderType, std::string>;

struct Workbody {
    utils::vkDefault::Pipeline pipeline;
    utils::vkDefault::PipelineLayout pipelineLayout;
    utils::vkDefault::DescriptorSetLayout descriptorSetLayout;

    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::DescriptorSets descriptorSets;

    virtual ~Workbody(){};
    virtual void create(const ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) = 0;
};

class Workflow
{
protected:
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    bool created{false};

    utils::vkDefault::RenderPass renderPass;
    utils::vkDefault::Framebuffers framebuffers;
    utils::vkDefault::CommandBuffers commandBuffers;

    virtual void updateCommandBuffer(uint32_t frameNumber) = 0;

public:
    virtual ~Workflow(){};

    Workflow& setDeviceProp(VkPhysicalDevice physicalDevice, VkDevice device);

    virtual void create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) = 0;
    virtual void updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) = 0;

    void update(uint32_t frameNumber);
    void raiseUpdateFlags();

    operator utils::vkDefault::CommandBuffers&();
    operator utils::vkDefault::CommandBuffers*();
};

struct Parameters {
    bool enable{false};
    utils::vkDefault::ImageInfo imageInfo;
    std::filesystem::path shadersPath;
};

using ParametersMap = std::unordered_map<std::string, workflows::Parameters*>;
using WorkflowsMap = std::unordered_map<std::string, std::unique_ptr<workflows::Workflow>>;

}
#endif // WORKFLOW_H
