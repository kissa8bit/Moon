#include "swapChain.h"
#include "buffer.h"

#include <cstring>

namespace moon::utils {

VkResult SwapChain::reset(const PhysicalDevice* pdevice, Window* pwindow, VkSurfaceKHR surfaceKHR, int32_t maxImageCount){
    device = pdevice;
    window = pwindow;
    surface = surfaceKHR;

    swapChain::SupportDetails supportDetails = swapChain::queryingSupport(*device, surface);
    VkSurfaceFormatKHR surfaceFormat = swapChain::queryingSurfaceFormat(supportDetails.formats);
    VkSurfaceCapabilitiesKHR capabilities = swapChain::queryingSupport(*device, surface).capabilities;

    imageInfo.Count = swapChain::queryingSupportImageCount(*device, surface);
    imageInfo.Count = (maxImageCount > 0 && imageInfo.Count > static_cast<uint32_t>(maxImageCount)) ? static_cast<uint32_t>(maxImageCount) : imageInfo.Count;
    imageInfo.Extent = swapChain::queryingExtent(window, capabilities);
    imageInfo.Format = surfaceFormat.format;

    std::vector<uint32_t> queueFamilyIndices = { 0 };
    VkResult result = swapChainKHR.reset(device->device(), imageInfo, supportDetails, queueFamilyIndices, surface, surfaceFormat);

    attachments.clear();
    for (const auto& image: swapChainKHR.images()){
        attachments.emplace_back();
        auto& attachment = attachments.back();
        attachment.image = image;
        attachment.imageView = utils::vkDefault::ImageView(device->device(), attachment.image, VK_IMAGE_VIEW_TYPE_2D, surfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1);
    }
    commandPool = utils::vkDefault::CommandPool(device->device());

    return result;
}

VkResult SwapChain::present(VkSemaphore waitSemaphore, ImageIndex imageIndex) const {
    const VkSwapchainKHR sc = swapChainKHR;
    const uint32_t idx = static_cast<uint32_t>(imageIndex);

    VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &waitSemaphore;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &sc;
        presentInfo.pImageIndices = &idx;

    return vkQueuePresentKHR(device->device()(0, 0), &presentInfo);
}

SwapChain::operator const VkSwapchainKHR&() const {
    return swapChainKHR;
}

const VkImageView& SwapChain::imageView(ImageIndex imageIndex) const {
    return attachments[static_cast<uint32_t>(imageIndex)].imageView;
}

utils::vkDefault::ImageInfo SwapChain::info() const { return imageInfo;}
VkSurfaceKHR SwapChain::getSurface() const { return surface;}
Window* SwapChain::getWindow() const { return window;}

VkResult SwapChain::acquireNextImage(VkSemaphore semaphore, ImageIndex& imageIndex) const
{
    uint32_t u32ImageIndex{};
    VkResult res = vkAcquireNextImageKHR(device->device(), swapChainKHR, UINT64_MAX, semaphore, VK_NULL_HANDLE, &u32ImageIndex);
    imageIndex = ImageIndex(u32ImageIndex);
    return res;
}

std::vector<uint32_t> SwapChain::makeScreenshot(ImageIndex imageIndex) const {
    std::vector<uint32_t> buffer(imageInfo.Extent.height * imageInfo.Extent.width, 0);

    Buffer cache(*device, device->device(), sizeof(uint32_t) * imageInfo.Extent.width * imageInfo.Extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device->device(), commandPool);
    texture::transitionLayout(commandBuffer, attachments.at(static_cast<uint32_t>(imageIndex)).image, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    texture::copy(commandBuffer, attachments.at(static_cast<uint32_t>(imageIndex)).image, cache, { imageInfo.Extent.width, imageInfo.Extent.height, 1}, 1);
    texture::transitionLayout(commandBuffer, attachments.at(static_cast<uint32_t>(imageIndex)).image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_REMAINING_MIP_LEVELS, 0, 1);
    singleCommandBuffer::submit(device->device(), device->device()(0,0), commandPool, &commandBuffer);

    CHECK_M(cache.map() != nullptr, std::string("[ SwapChain::makeScreenshot ] cache buffer not mapped"));
    std::memcpy(buffer.data(), cache.map(), cache.size());

    return buffer;
}

}
