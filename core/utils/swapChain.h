#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include <vulkan.h>

#include "attachments.h"
#include "operations.h"
#include "device.h"

namespace moon::utils {

class SwapChain{
private:
    struct SwapChainAttachment {
        VkImage image{ VK_NULL_HANDLE };
        utils::vkDefault::ImageView imageView;
    };

    GLFWwindow* window{ nullptr };
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

    VkResult reset(const PhysicalDevice* device, GLFWwindow* window, VkSurfaceKHR surface, int32_t maxImageCount = -1);
    VkResult present(VkSemaphore waitSemaphore, uint32_t imageIndex) const;

    operator const VkSwapchainKHR&() const;
    const VkImageView& SwapChain::imageView(uint32_t i) const;

    utils::vkDefault::ImageInfo info() const;
    VkSurfaceKHR getSurface() const;
    GLFWwindow* getWindow() const;

    std::vector<uint32_t> makeScreenshot(uint32_t i = 0) const;
};

}
#endif // SWAPCHAIN_H
