#include "graphicsLinker.h"

#include <utils/attachments.h>
#include <utils/swapChain.h>

namespace moon::graphicsManager {

GraphicsLinker::GraphicsLinker(GraphicsLinker&& other) {
    swap(other);
}

GraphicsLinker& GraphicsLinker::operator=(GraphicsLinker&& other) {
    swap(other);
    return *this;
}

void GraphicsLinker::swap(GraphicsLinker& other) {
    std::swap(graphics, other.graphics);
    std::swap(imageInfo, other.imageInfo);
    renderPass.swap(other.renderPass);
    framebuffers.swap(other.framebuffers);
    commandPool.swap(other.commandPool);
    commandBuffers.swap(other.commandBuffers);
    signalSemaphores.swap(other.signalSemaphores);
}

GraphicsLinker::GraphicsLinker(VkDevice device, const moon::utils::SwapChain* swapChainKHR, std::vector<GraphicsInterface*>* graphics)
    : graphics(graphics), imageInfo(swapChainKHR->info()) 
{
    createRenderPass(device);
    createFramebuffers(device, swapChainKHR);
    createCommandBuffers(device);
    createSyncObjects(device);

    for (auto& graph : *graphics) {
        graph->link->renderPass() = renderPass;
    }
}

void GraphicsLinker::createRenderPass(VkDevice device) {
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(imageInfo.Format, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    };

    utils::vkDefault::SubpassInfos subpassInfos = utils::vkDefault::subpassInfos(attachments.size());

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
    dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies.back().dstSubpass = 0;
    dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, dependencies);
}

void GraphicsLinker::createFramebuffers(VkDevice device, const moon::utils::SwapChain* swapChainKHR){
    framebuffers.resize(imageInfo.Count);
    for (uint32_t i = 0; i < framebuffers.size(); i++) {
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &swapChainKHR->imageView(utils::ImageIndex(i));
            framebufferInfo.width = imageInfo.Extent.width;
            framebufferInfo.height = imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers.at(i) = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void GraphicsLinker::createCommandBuffers(VkDevice device){
    commandPool = utils::vkDefault::CommandPool(device);
    commandBuffers = commandPool.allocateCommandBuffers(imageInfo.Count);
}

void GraphicsLinker::createSyncObjects(VkDevice device) {
    signalSemaphores.resize(imageInfo.Count);
    for (auto& semaphore : signalSemaphores) {
        semaphore = utils::vkDefault::Semaphore(device);
    }
}

void GraphicsLinker::update(utils::ResourceIndex resourceIndex, utils::ImageIndex imageIndex){
    CHECK(commandBuffers.at(imageIndex).reset());
    CHECK(commandBuffers.at(imageIndex).begin());

    std::vector<VkClearValue> clearValues = {VkClearValue{}};
    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers.at(imageIndex);
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers.at(imageIndex), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for(auto& graph: *graphics){
        graph->link->draw(commandBuffers.at(imageIndex), resourceIndex);
    }

    vkCmdEndRenderPass(commandBuffers.at(imageIndex));

    CHECK(commandBuffers.at(imageIndex).end());
}

VkRenderPass GraphicsLinker::getRenderPass() const {
    return renderPass;
}

VkSemaphore GraphicsLinker::submit(utils::ImageIndex imageIndex, const utils::vkDefault::VkSemaphores& waitSemaphores, VkFence fence, VkQueue queue) const  {
    VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = submitInfo.waitSemaphoreCount > 0 ? waitSemaphores.data() : VK_NULL_HANDLE;
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = commandBuffers.at(imageIndex);
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores.at(imageIndex);
    CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    return signalSemaphores.at(imageIndex);
}

} // moon::graphicsManager
