#ifndef NODE_H
#define NODE_H

#include <vector>

#include <vulkan.h>
#include <vkdefault.h>

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
    VkResult submit(uint32_t frameIndex) const;
};

using PipelineStages = std::vector<PipelineStage>;

class PipelineNode{
private:
    PipelineStages stages;
    PipelineNode* next{nullptr};

    vkDefault::VkSemaphores semaphores(uint32_t frameIndex);

public:
    PipelineNode() = default;
    PipelineNode(VkDevice device, PipelineStages&& stages, PipelineNode* next = nullptr);

    vkDefault::VkSemaphores submit(const uint32_t frameIndex, const vkDefault::VkSemaphores& externalSemaphore = {});
};

using PipelineNodes = std::vector<PipelineNode>;

}
#endif // NODE_H
