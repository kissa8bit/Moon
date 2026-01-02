#include "pipelineNode.h"
#include "operations.h"

#include <unordered_set>

namespace moon::utils {

PipelineStage::PipelineStage(const std::vector<const vkDefault::CommandBuffers*>& commandBuffers, VkPipelineStageFlags waitStagesMask, VkQueue queue)
    : waitStagesMask(waitStagesMask), queue(queue)
{
    std::unordered_set<size_t> sizes;
    for (const auto& fcb : commandBuffers) {
        sizes.insert(fcb->size());
    }

    if (!CHECK_M(sizes.size() == 1, std::string("[PipelineStage::PipelineStage] input commandBuffers must be same size"))) return;

    const size_t frameCount = *sizes.begin();
    frames.resize(frameCount);

    for (size_t idx = 0; idx < frameCount; ++idx) {
        for (const auto& fcb : commandBuffers) {
            frames[idx].commandBuffers.push_back((*fcb)[idx]);
        }
    }
}

VkResult PipelineStage::submit(uint32_t frameIndex) const {
    const auto& frame = frames.at(frameIndex);

    const uint32_t waitCount = static_cast<uint32_t>(frame.wait.size());
    std::vector<VkPipelineStageFlags> waitStagesMasks;
    const VkPipelineStageFlags* pWaitDstStageMask = nullptr;
    const VkSemaphore* pWaitSemaphores = nullptr;
    if (waitCount > 0) {
        waitStagesMasks.assign(waitCount, waitStagesMask);
        pWaitDstStageMask = waitStagesMasks.data();
        pWaitSemaphores = frame.wait.data();
    }

    std::vector<VkSemaphore> signalSemaphores;
    signalSemaphores.reserve(frame.signal.size());
    for (const auto& sem : frame.signal) {
        signalSemaphores.push_back(sem);
    }
    const uint32_t signalCount = static_cast<uint32_t>(signalSemaphores.size());
    const VkSemaphore* pSignalSemaphores = signalCount > 0 ? signalSemaphores.data() : nullptr;

    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = waitCount;
        submitInfo.pWaitSemaphores = pWaitSemaphores;
        submitInfo.pWaitDstStageMask = pWaitDstStageMask;
        submitInfo.commandBufferCount = static_cast<uint32_t>(frame.commandBuffers.size());
        submitInfo.pCommandBuffers = frame.commandBuffers.data();
        submitInfo.signalSemaphoreCount = signalCount;
        submitInfo.pSignalSemaphores = pSignalSemaphores;
    return vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
}

PipelineNode::PipelineNode(VkDevice device, PipelineStages&& instages, PipelineNode* next) : stages(std::move(instages)), next(next) {
    for (auto& currentStage : stages) {
        for (size_t frameIndex = 0; frameIndex < currentStage.frames.size(); ++frameIndex) {
            auto& frame = currentStage.frames[frameIndex];
            if (next) {
                for (auto& nextStage : next->stages) {
                    frame.signal.emplace_back(utils::vkDefault::Semaphore(device));
                    const auto& signal = frame.signal.back();
                    nextStage.frames.at(frameIndex).wait.push_back(signal);
                }
            } else {
                frame.signal.emplace_back(utils::vkDefault::Semaphore(device));
            }
        }
    }
}

vkDefault::VkSemaphores PipelineNode::submit(const uint32_t frameIndex, const vkDefault::VkSemaphores& externalSemaphore){
    if (!externalSemaphore.empty()) {
        CHECK_M(stages.size() == 1, std::string("[PipelineStage::submit] first PipelineNode must have single PipelineStage"));
        stages.front().frames.at(frameIndex).wait = externalSemaphore;
    }

    for(const auto& stage: stages){
        CHECK(stage.submit(frameIndex));
    }

    return next ? next->submit(frameIndex) : semaphores(frameIndex);
}

vkDefault::VkSemaphores PipelineNode::semaphores(uint32_t frameIndex) {
    vkDefault::VkSemaphores semaphores;
    for (const auto& stage : stages) {
        const auto& signal = stage.frames.at(frameIndex).signal;
        CHECK_M(signal.size() == 1, std::string("[PipelineStage::semaphores] each final PipelineStage must have single signal semaphore"));
        semaphores.push_back(signal.front());
    }
    return semaphores;
}

}
