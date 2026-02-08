#ifndef MOON_GRAPHICS_MANAGER_GRAPHICSLINKER_H
#define MOON_GRAPHICS_MANAGER_GRAPHICSLINKER_H

#include <vector>
#include <stdint.h>

#include <vulkan.h>

#include <utils/vkdefault.h>
#include <utils/swapChain.h>
#include <utils/types.h>

#include "graphicsInterface.h"

namespace moon::graphicsManager {

class GraphicsLinker
{
private:
    std::vector<GraphicsInterface*>*    graphics;
    moon::utils::vkDefault::ImageInfo   imageInfo;

    utils::vkDefault::RenderPass     renderPass;
    utils::vkDefault::Framebuffers   framebuffers;
    utils::vkDefault::CommandPool    commandPool;
    utils::vkDefault::CommandBuffers commandBuffers;
    utils::vkDefault::Semaphores     signalSemaphores;

    void createRenderPass(VkDevice device);
    void createFramebuffers(VkDevice device, const moon::utils::SwapChain* swapChainKHR);
    void createCommandBuffers(VkDevice device);
    void createSyncObjects(VkDevice device);

public:
    GraphicsLinker() = default;
    GraphicsLinker(const GraphicsLinker&) = delete;
    GraphicsLinker& operator=(const GraphicsLinker&) = delete;
    GraphicsLinker(GraphicsLinker&&);
    GraphicsLinker& operator=(GraphicsLinker&&);
    void swap(GraphicsLinker&);

    GraphicsLinker(VkDevice device, const moon::utils::SwapChain* swapChainKHR, std::vector<GraphicsInterface*>* graphics);
    void update(utils::ResourceIndex resourceIndex, utils::ImageIndex imageNumber);
    VkRenderPass getRenderPass() const;

    VkSemaphore submit(utils::ImageIndex imageIndex, const utils::vkDefault::VkSemaphores& waitSemaphores, VkFence fence, VkQueue queue) const;
};

} // moon::graphicsManager

#endif // MOON_GRAPHICS_MANAGER_GRAPHICSLINKER_H
