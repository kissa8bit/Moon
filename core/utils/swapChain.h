#ifndef MOON_UTILS_SWAPCHAIN_H
#define MOON_UTILS_SWAPCHAIN_H

#include <vulkan.h>

#include "attachments.h"
#include "operations.h"
#include "device.h"
#include "types.h"

namespace moon::utils {

class SwapChain{
private:
    struct SwapChainAttachment {
        VkImage image{ VK_NULL_HANDLE };
        utils::vkDefault::ImageView imageView;
    };

    Window* window{ nullptr };
    const PhysicalDevice* device{ nullptr };
    VkSurfaceKHR surface{ VK_NULL_HANDLE };
    utils::vkDefault::ImageInfo imageInfo;

    utils::vkDefault::SwapchainKHR swapChainKHR;
    utils::vkDefault::CommandPool commandPool;
    std::vector<SwapChainAttachment> attachments;

public:
    SwapChain() = default;
    SwapChain(const SwapChain&) = delete;
    SwapChain& operator=(const SwapChain&) = delete;
    SwapChain(SwapChain&&) = delete;
    SwapChain& operator=(SwapChain&&) = delete;

    VkResult reset(const PhysicalDevice* device, Window* window, VkSurfaceKHR surface, int32_t maxImageCount = -1);
    VkResult present(VkSemaphore waitSemaphore, ImageIndex imageIndex) const;

    operator const VkSwapchainKHR&() const;
    const VkImageView& SwapChain::imageView(ImageIndex imageIndex) const;

    utils::vkDefault::ImageInfo info() const;
    VkSurfaceKHR getSurface() const;
    Window* getWindow() const;

    VkResult acquireNextImage(VkSemaphore semaphore, ImageIndex& imageIndex) const;

    std::vector<uint32_t> makeScreenshot(ImageIndex imageIndex = ImageIndex(0)) const;
};

}
#endif // MOON_UTILS_SWAPCHAIN_H
