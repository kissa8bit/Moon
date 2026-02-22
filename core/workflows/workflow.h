#ifndef MOON_WORKFLOWS_WORKFLOW_H
#define MOON_WORKFLOWS_WORKFLOW_H

#include <filesystem>
#include <unordered_map>
#include <memory>

#include <vulkan.h>

#include <utils/attachments.h>
#include <utils/buffer.h>
#include <utils/vkdefault.h>

namespace moon::workflows {

enum class ShaderType
{
    Vertex,
    Fragment,
    Compute
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

struct ParameterNameTag {};
struct WorkflowNameTag {}; 

using ParameterName = utils::Name<ParameterNameTag>;
using WorkflowName = utils::Name<WorkflowNameTag>;

using ParametersMap = std::unordered_map<ParameterName, Parameters*>;
using WorkflowsMap = std::unordered_map<WorkflowName, std::unique_ptr<workflows::Workflow>>;

} // moon::workflows

#endif // MOON_WORKFLOWS_WORKFLOW_H
