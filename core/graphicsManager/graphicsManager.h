#ifndef GRAPHICSMANAGER_GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_GRAPHICSMANAGER_H

#include <vulkan.h>

#include <utils/operations.h>
#include <utils/device.h>
#include <utils/swapChain.h>
#include <utils/types.h>

#include "graphicsInterface.h"
#include "graphicsLinker.h"

namespace moon::graphicsManager {

class GraphicsManager
{
private:
    const std::vector<const char*>          validationLayers{"VK_LAYER_KHRONOS_validation"};
    const std::vector<std::string>          deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    utils::vkDefault::Instance              instance;
    utils::vkDefault::DebugUtilsMessenger   debugMessenger;
    utils::vkDefault::Surface               surface;

    moon::utils::PhysicalDevice::Map        devices;
    moon::utils::PhysicalDevice*            activeDevice{nullptr};
    moon::utils::SwapChain                  swapChainKHR;

    std::vector<GraphicsInterface*>         graphics;
    GraphicsLinker                          linker;

    utils::vkDefault::Semaphores            availableSemaphores;
    utils::vkDefault::Fences                fences;

    utils::ImageIndex imageIndex{0};
    uint32_t imageCount{0};
    utils::ResourceIndex resourceIndex{0};
    uint32_t resourceCount{0};

    VkResult createInstance(utils::Window* window);
    VkResult createDevice(const VkPhysicalDeviceFeatures& deviceFeatures = {});
    VkResult createSurface(utils::Window* window);
    VkResult createSwapChain(utils::Window * window, int32_t maxImageCount = -1);
    VkResult createLinker();
    VkResult createSyncObjects();

public:
    GraphicsManager(utils::Window* window, uint32_t imageCount = 0, uint32_t resourceCount = 1, const VkPhysicalDeviceFeatures& deviceFeatures = {});

    VkInstance              getInstance() const;
    VkExtent2D              getImageExtent() const;
    utils::ResourceIndex    getResourceIndex() const;
    uint32_t                getResourceCount() const;
    utils::ImageIndex       getImageIndex() const;
    uint32_t                getImageCount() const;

    void setDevice(utils::PhysicalDevice::Index deviceIndex);
    void setGraphics(GraphicsInterface* graphics);

    void reset(utils::Window* window);

    VkResult checkNextFrame();
    VkResult drawFrame();
    VkResult deviceWaitIdle() const;

    std::vector<uint32_t> makeScreenshot() const;
};

} // moon::graphicsManager

#endif // GRAPHICSMANAGER_GRAPHICSMANAGER_H
