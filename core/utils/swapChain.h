#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include <vulkan.h>
#include <vector>

#include "attachments.h"
#include "operations.h"

class swapChain
{
private:
    VkPhysicalDevice    physicalDevice{VK_NULL_HANDLE};
    VkDevice            device{VK_NULL_HANDLE};
    uint32_t            imageCount{0};

    VkSwapchainKHR      swapChainKHR{VK_NULL_HANDLE};
    attachments         swapChainAttachments;

public:
    swapChain();
    void destroy();

    void create(GLFWwindow* window, VkSurfaceKHR* surface, uint32_t queueFamilyIndexCount, uint32_t* pQueueFamilyIndices, int32_t maxImageCount = -1);

    VkSwapchainKHR& operator()();
    attachments& attachment();
    void setDevice(VkPhysicalDevice physicalDevice, VkDevice device);
    uint32_t getImageCount();
};

#endif // SWAPCHAIN_H