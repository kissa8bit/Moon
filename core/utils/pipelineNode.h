#ifndef MOON_UTILS_NODE_H
#define MOON_UTILS_NODE_H

#include <vector>

#include <vulkan.h>

#include "vkdefault.h"
#include "types.h"

namespace moon::utils {

struct PipelineStage{
    struct Frame {
        std::vector<VkCommandBuffer> commandBuffers;
        vkDefault::VkSemaphores wait;
        vkDefault::Semaphores signal;
    };
    std::vector<Frame> frames;

    VkPipelineStageFlags waitStagesMask{};
    VkQueue queue{VK_NULL_HANDLE};

    PipelineStage(const std::vector<const vkDefault::CommandBuffers*>& commandBuffers, VkPipelineStageFlags waitStagesMask, VkQueue queue);
    VkResult submit(ResourceIndex resourceIndex) const;
};

using PipelineStages = std::vector<PipelineStage>;

class PipelineNode{
private:
    PipelineStages stages;
    PipelineNode* next{nullptr};

    vkDefault::VkSemaphores semaphores(ResourceIndex resourceIndex);

public:
    PipelineNode() = default;
    PipelineNode(VkDevice device, PipelineStages&& stages, PipelineNode* next = nullptr);

    vkDefault::VkSemaphores submit(ResourceIndex resourceIndex, const vkDefault::VkSemaphores& externalSemaphore = {});
};

using PipelineNodes = std::vector<PipelineNode>;

}
#endif // MOON_UTILS_NODE_H
