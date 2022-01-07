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

    frames.resize(*sizes.begin());
    for (const auto& fcb : commandBuffers) {
        for (const auto& cb : *fcb) {
            frames.at(&cb - &fcb->front()).commandBuffers.push_back(cb);
        }
    }
}

VkResult PipelineStage::submit(uint32_t frameIndex) const {
    const auto& frame = frames.at(frameIndex);
    std::vector<VkPipelineStageFlags> waitStagesMasks(frame.wait.size(), waitStagesMask);
    vkDefault::VkSemaphores signals(frame.signal.begin(), frame.signal.end());
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(frame.wait.size());
        submitInfo.pWaitSemaphores = frame.wait.data();
        submitInfo.pWaitDstStageMask = waitStagesMasks.data();
        submitInfo.commandBufferCount = static_cast<uint32_t>(frame.commandBuffers.size());
        submitInfo.pCommandBuffers = frame.commandBuffers.data();
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
        submitInfo.pSignalSemaphores = signals.data();
    return vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
}

PipelineNode::PipelineNode(VkDevice device, PipelineStages&& instages, PipelineNode* next) : stages(std::move(instages)), next(next) {
    for (auto& currentStage : stages) {
        for (auto& frame : currentStage.frames) {
            if (next) {
                const auto frameIndex = &frame - &currentStage.frames.front();
                for (auto& nextStage : next->stages) {
                    const auto& signal = frame.signal.emplace_back(utils::vkDefault::Semaphore(device));
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
