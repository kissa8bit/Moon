#include "operations.h"

#include "memory.h"

#include <vulkan/vk_enum_string_helper.h>

#include <set>
#include <utility>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <unordered_map>

#define ONLYDEVICELOCALHEAP
#define THROW_EXEPTION

namespace moon::utils {

void debug::displayError(std::string message) {
#ifndef DEBUG_PRINT_DISABLE
    std::cerr << message << std::endl;
#endif
#ifdef THROW_EXEPTION
    throw std::runtime_error(message);
#endif
}

VkResult debug::checkResult(VkResult result, std::string message) {
    if (result != VK_SUCCESS){
        debug::displayError(string_VkResult(result) + std::string(" : ") + message);
    }
    return result;
}

bool debug::checkResult(bool result, std::string message) {
    if (!result){
        debug::displayError(message);
    }
    return result;
}

bool validationLayer::checkSupport(const std::vector<const char*>& validationLayers) {
    uint32_t layerCount;
    CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));

    std::vector<VkLayerProperties> availableLayers(layerCount);
    CHECK(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

    bool res = true;
    for (const auto& layerName : validationLayers){
        bool layerFound = false;
        for(const auto& layerProperties: availableLayers){
            layerFound |= (strcmp(layerName, layerProperties.layerName) == 0);
        }
        res &= layerFound;
    }
    return res;
}

void validationLayer::destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr) func(instance, debugMessenger, pAllocator);
}

void validationLayer::setupDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT* debugMessenger) {
    if(auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"); func != nullptr){
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = debugCallback;
        func(instance, &createInfo, nullptr, debugMessenger);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL validationLayer::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

void physicalDevice::printMemoryProperties(VkPhysicalDeviceMemoryProperties memoryProperties){
    std::cout << "memoryHeapCount = " << memoryProperties.memoryHeapCount << std::endl;
    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++){
        std::cout << "heapFlag[" << i << "] = " << memoryProperties.memoryHeaps[i].flags << "\t\t"
                  << "heapSize[" << i << "] = " << memoryProperties.memoryHeaps[i].size << std::endl;
    }
    std::cout << "memoryTypeCount = " << memoryProperties.memoryTypeCount << std::endl;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
        std::cout << "heapIndex[" << i << "] = " << memoryProperties.memoryTypes[i].heapIndex << '\t'
                  << "heapType [" << i << "] = " << memoryProperties.memoryTypes[i].propertyFlags << std::endl;
    }
    std::cout<<std::endl;
}

uint32_t physicalDevice::findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

#ifdef ONLYDEVICELOCALHEAP
    uint32_t deviceLocalHeapIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryHeapCount; i++){
        if(memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT){
            deviceLocalHeapIndex = i;
            break;
        }
    }
#endif

    std::vector<uint32_t> memoryTypeIndex;

#ifdef ONLYDEVICELOCALHEAP
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
        if ((memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
            if(memoryProperties.memoryTypes[i].heapIndex == deviceLocalHeapIndex){
                memoryTypeIndex.push_back(i);
			}
        }
    }
#endif

    if(memoryTypeIndex.size() == 0){
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++){
		    if ((memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
				memoryTypeIndex.push_back(i);
		    }
    	}
	}
    return memoryTypeIndex.size() != 0 ? memoryTypeIndex[0] : UINT32_MAX;
}

void physicalDevice::printQueueIndices(VkPhysicalDevice device, VkSurfaceKHR surface) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t index = 0; index < queueFamilyCount; index++){
        VkBool32 presentSupport = false;
        if(surface){
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport));
        }

        std::cout << "index\t" << index << "\tqueue count\t" << queueFamilies[index].queueCount << std::endl;
        if(queueFamilies[index].queueFlags & VK_QUEUE_GRAPHICS_BIT){
            std::cout << "index\t" << index << "\t:\t" << "VK_QUEUE_GRAPHICS_BIT" << std::endl;
        }
        if(queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT){
            std::cout << "index\t" << index << "\t:\t" << "VK_QUEUE_COMPUTE_BIT" << std::endl;
        }
        if(queueFamilies[index].queueFlags & VK_QUEUE_TRANSFER_BIT){
            std::cout << "index\t" << index << "\t:\t" << "VK_QUEUE_TRANSFER_BIT" << std::endl;
        }
        if(queueFamilies[index].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT){
            std::cout << "index\t" << index << "\t:\t" << "VK_QUEUE_SPARSE_BINDING_BIT" << std::endl;
        }
        if(queueFamilies[index].queueFlags & VK_QUEUE_PROTECTED_BIT){
            std::cout << "index\t" << index << "\t:\t" << "VK_QUEUE_PROTECTED_BIT" << std::endl;
        }
        if(presentSupport){
            std::cout << "index\t" << index << "\t:\t" << "Present Support" << std::endl;
        }
        std::cout << std::endl;
    }
}

std::vector<uint32_t> physicalDevice::findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::vector<uint32_t> indices;

    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, queueFamilies.data());

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++){
        VkBool32 presentSupport = surface ? false : true;
        if(surface){
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport));
        }
        if (presentSupport){
            indices.push_back(index);
        }
    }

    return indices;
}

std::vector<uint32_t> physicalDevice::findQueueFamilies(VkPhysicalDevice device, VkQueueFlagBits queueFlags, VkSurfaceKHR surface) {
    std::vector<uint32_t> indices;

    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, queueFamilies.data());

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++){
        VkBool32 presentSupport = surface ? false : true;
        if(surface){
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport));
        }
        if ((queueFamilies[index].queueFlags & queueFlags) == queueFlags && presentSupport){
            indices.push_back(index);
        }
    }

    return indices;
}

std::vector<VkQueueFamilyProperties> physicalDevice::findQueueFamiliesProperties(VkPhysicalDevice device, VkQueueFlagBits queueFlags, VkSurfaceKHR surface) {
    std::vector<VkQueueFamilyProperties> result;

    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertyCount, queueFamilies.data());

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++){
        VkBool32 presentSupport = surface ? false : true;
        if(surface){
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport));
        }
        if ((queueFamilies[index].queueFlags & queueFlags) == queueFlags && presentSupport){
            result.push_back(queueFamilies[index]);
        }
    }

    return result;
}

VkSampleCountFlagBits physicalDevice::queryingSampleCount(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT)  { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT)  { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT)  { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

bool physicalDevice::isSuitable(VkPhysicalDevice device, VkSurfaceKHR surface, const std::vector<std::string>& deviceExtensions) {
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return physicalDevice::isExtensionsSupport(device, deviceExtensions) && swapChain::queryingSupport(device, surface).isNotEmpty() && supportedFeatures.samplerAnisotropy;
}

bool physicalDevice::isExtensionsSupport(VkPhysicalDevice device, const std::vector<std::string>& deviceExtensions) {
    uint32_t extensionCount;
    CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,nullptr));

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,availableExtensions.data()));

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

VkResult buffer::create(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* bufferMemory) {
    VkResult result = VK_SUCCESS;

    VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(result = vkCreateBuffer(device, &bufferInfo, nullptr, buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memoryRequirements);

    CHECK(result = Memory::instance().allocate(physicalDevice, device, memoryRequirements, properties, bufferMemory));
    CHECK(result = vkBindBufferMemory(device, *buffer, *bufferMemory, 0));

    return result;
}

void buffer::copy(VkCommandBuffer commandBuffer, VkDeviceSize size, VkBuffer srcBuffer, VkBuffer dstBuffer) {
    VkBufferCopy copyRegion{};
        copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
}

void buffer::destroy(VkDevice device, VkBuffer& buffer, VkDeviceMemory& memory) {
    if(buffer){
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    if(memory){
        Memory::instance().free(memory);
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

VkCommandBuffer singleCommandBuffer::create(VkDevice device, VkCommandPool commandPool) {
    VkResult result = VK_SUCCESS;
    VkCommandBuffer commandBuffer;

    VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;
    CHECK(result = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));


    VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
    CHECK(result = vkBeginCommandBuffer(commandBuffer, &beginInfo));

    return commandBuffer;
}

VkResult singleCommandBuffer::submit(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer* commandBuffer) {
    return singleCommandBuffer::submit(device, queue, commandPool, 1, commandBuffer);
}

VkResult singleCommandBuffer::submit(VkDevice device, VkQueue queue, VkCommandPool commandPool, uint32_t commandBufferCount, VkCommandBuffer* commandBuffer) {
    VkResult result = VK_SUCCESS;
    for(size_t i = 0; i < commandBufferCount; i++){
        CHECK(result = vkEndCommandBuffer(commandBuffer[i]));
    }

    VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = commandBufferCount;
        submitInfo.pCommandBuffers = commandBuffer;
    CHECK(result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    CHECK(result = vkQueueWaitIdle(queue));

    vkFreeCommandBuffers(device, commandPool, commandBufferCount, commandBuffer);

    return result;
}

singleCommandBuffer::Scoped::Scoped(VkDevice device, VkQueue queue, VkCommandPool commandPool):
    commandBuffer(create(device, commandPool)), device(device), queue(queue), commandPool(commandPool)
{}

singleCommandBuffer::Scoped::~Scoped() {
    if(commandBuffer != VK_NULL_HANDLE){
        submit(device, queue, commandPool, &commandBuffer);
    }
}

singleCommandBuffer::Scoped::operator VkCommandBuffer() {
    return commandBuffer;
}

void texture::transitionLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t arrayLayers){
    std::unordered_map<VkImageLayout,std::pair<VkAccessFlags,VkPipelineStageFlags>> layoutDescription = {
        {VK_IMAGE_LAYOUT_UNDEFINED, {0,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT}},
        {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, {VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT}},
        {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, {VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT}},
        {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT}},
        {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, {VK_ACCESS_TRANSFER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT}}
    };

    auto getDesc = [&layoutDescription](VkImageLayout layout) -> std::pair<VkAccessFlags, VkPipelineStageFlags> {
        if (auto it = layoutDescription.find(layout); it != layoutDescription.end()) {
            return it->second;
        }
        return { 0u, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };
    };

    const auto srcDesc = getDesc(oldLayout);
    const auto dstDesc = getDesc(newLayout);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    barrier.subresourceRange.layerCount = arrayLayers;
    barrier.srcAccessMask = srcDesc.first;
    barrier.dstAccessMask = dstDesc.first;
    vkCmdPipelineBarrier(commandBuffer, srcDesc.second, dstDesc.second, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void texture::copy(VkCommandBuffer commandBuffer, VkImage srcImage, VkImage dstImage, VkExtent3D extent, uint32_t layerCount) {
    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount = layerCount;
    region.srcOffset = { 0, 0, 0 };
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount = layerCount;
    region.dstOffset = { 0, 0, 0 };
    region.extent = extent;
    vkCmdCopyImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void texture::copy(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkExtent3D extent, uint32_t layerCount){
    VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = extent;
    vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void texture::copy(VkCommandBuffer commandBuffer, VkImage srcImage, VkBuffer dstBuffer, VkExtent3D extent, uint32_t layerCount){
    VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = extent;
    vkCmdCopyImageToBuffer(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer, 1, &region);
}

VkResult texture::create(VkPhysicalDevice physicalDevice, VkDevice device, VkImageCreateFlags flags, VkExtent3D extent, uint32_t arrayLayers, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageLayout layout, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage* image, VkDeviceMemory* imageMemory){
    VkResult result = VK_SUCCESS;

    VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.flags = flags;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = extent;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = arrayLayers;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = layout;
        imageInfo.usage = usage;
        imageInfo.samples = numSamples;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(result = vkCreateImage(device, &imageInfo, nullptr, image));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, *image, &memRequirements);

    CHECK(result = Memory::instance().allocate(physicalDevice, device, memRequirements, properties, imageMemory));
    CHECK(result = vkBindImageMemory(device, *image, *imageMemory, 0));

    return result;
}

void texture::destroy(VkDevice device, VkImage& image, VkDeviceMemory& memory) {
    if(image) {
        vkDestroyImage(device, image, nullptr);
        image = VK_NULL_HANDLE;
    }
    if(memory) {
        Memory::instance().free(memory);
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

VkResult texture::createView(VkDevice device, VkImageViewType type, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount, VkImage image, VkImageView* imageView) {
    VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = type;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = baseArrayLayer;
        viewInfo.subresourceRange.layerCount = layerCount;
    return vkCreateImageView(device, &viewInfo, nullptr, imageView);
}

void texture::generateMipmaps(VkPhysicalDevice physicalDevice, VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount) {
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)){
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;
        barrier.subresourceRange.levelCount = 1;

    for (uint32_t i = 1, mipWidth = texWidth, mipHeight = texHeight; i < mipLevels; i++, mipWidth /= 2, mipHeight /= 2) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        blitDown(commandBuffer,image,i - 1,image,i,(mipWidth > 1 ? mipWidth : 1),(mipHeight > 1 ? mipHeight : 1),baseArrayLayer,layerCount,2);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void texture::blitDown(VkCommandBuffer commandBuffer, VkImage srcImage, uint32_t srcMipLevel, VkImage dstImage, uint32_t dstMipLevel, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount, float blitFactor) {
    VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width),static_cast<int32_t>(height),1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = srcMipLevel;
        blit.srcSubresource.baseArrayLayer = baseArrayLayer;
        blit.srcSubresource.layerCount = layerCount;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {
            static_cast<int32_t>((width / blitFactor) > 1.0f ? static_cast<int32_t>(width / blitFactor) : 1),
            static_cast<int32_t>((height / blitFactor) > 1.0f ? static_cast<int32_t>(height / blitFactor) : 1),
            1
        }; 
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = dstMipLevel;
        blit.dstSubresource.baseArrayLayer = baseArrayLayer;
        blit.dstSubresource.layerCount = layerCount;
    vkCmdBlitImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}

void texture::blitUp(VkCommandBuffer commandBuffer, VkImage srcImage, uint32_t srcMipLevel, VkImage dstImage, uint32_t dstMipLevel, uint32_t width, uint32_t height, uint32_t baseArrayLayer, uint32_t layerCount, float blitFactor) {
    VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width/blitFactor),static_cast<int32_t>(height/blitFactor),1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = srcMipLevel;
        blit.srcSubresource.baseArrayLayer = baseArrayLayer;
        blit.srcSubresource.layerCount = layerCount;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width),static_cast<int32_t>(height),1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = dstMipLevel;
        blit.dstSubresource.baseArrayLayer = baseArrayLayer;
        blit.dstSubresource.layerCount = layerCount;
    vkCmdBlitImage(commandBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
}

swapChain::SupportDetails swapChain::queryingSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    swapChain::SupportDetails details{};
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    CHECK(result);

    uint32_t formatCount = 0, presentModeCount = 0;

    result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    CHECK(result);
    if (formatCount != 0){
        details.formats.resize(formatCount);
        result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        CHECK(result);
    }

    result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    CHECK(result);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        CHECK(result);
    }

    return details;
}

uint32_t swapChain::queryingSupportImageCount(VkPhysicalDevice device, VkSurfaceKHR surface) {
    auto capabilities = swapChain::queryingSupport(device, surface).capabilities;
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount){
        imageCount = capabilities.maxImageCount;
    }
    return imageCount;
}

VkSurfaceFormatKHR swapChain::queryingSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR swapChain::queryingPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D swapChain::queryingExtent(Window* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if(!CHECK_M(window, "[ swapChain::queryingExtent ] : window is nullptr")) return{};
    const auto [width, height] = window->getFramebufferSize();

    VkExtent2D actualExtent;
    if (capabilities.currentExtent.width != UINT32_MAX && capabilities.currentExtent.height != UINT32_MAX) {
        actualExtent = capabilities.currentExtent;
    }
    else {
        actualExtent.width = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    return actualExtent;
}

VkFormat image::supportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    VkFormat supportedFormat = candidates[0];
    for (VkFormat format : candidates)
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if ((tiling == VK_IMAGE_TILING_OPTIMAL || tiling == VK_IMAGE_TILING_LINEAR) && (props.linearTilingFeatures & features) == features){
            supportedFormat = format; break;
        }
    }
    return supportedFormat;
}

std::vector<VkFormat> image::depthFormats() {
    return { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT };
}

VkFormat image::depthStencilFormat(VkPhysicalDevice physicalDevice)
{
    return image::supportedFormat(
        physicalDevice,
        image::depthFormats(),
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

std::vector<char> shaderModule::readFile(const std::filesystem::path& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!CHECK_M(file.is_open(), "[ shaderModule::readFile ] : failed to open file " + filename.string())) {
        return std::vector<char>{};
    }
    std::streampos pos = file.tellg();
    if (!CHECK_M(pos != std::streampos(-1), "[ shaderModule::readFile ] : failed to determine file size " + filename.string())) {
        file.close();
        return std::vector<char>{};
    }
    size_t fileSize = static_cast<size_t>(pos);
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule shaderModule::create(VkDevice device, const std::vector<char>& code) {
    CHECK_M((code.size() % 4) == 0, std::string("[ shaderModule::create ] shader code size not multiple of 4"));
    VkShaderModule shaderModule{VK_NULL_HANDLE};
    VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}

void descriptorSet::update(VkDevice device, const Writes& writes) {
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void descriptorSet::write(Writes& writes, VkDescriptorSet descriptorSet, const VkDescriptorBufferInfo& bufferInfo, VkDescriptorType descriptorType) {
    writes.push_back(VkWriteDescriptorSet{});
    writes.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.back().dstSet = descriptorSet;
    writes.back().dstBinding = static_cast<uint32_t>(writes.size() - 1);
    writes.back().dstArrayElement = 0;
    writes.back().descriptorType = descriptorType;
    writes.back().descriptorCount = 1;
    writes.back().pBufferInfo = &bufferInfo;
}

void descriptorSet::write(Writes& writes, VkDescriptorSet descriptorSet, const VkDescriptorImageInfo& imageInfo, VkDescriptorType descriptorType) {
    writes.push_back(VkWriteDescriptorSet{});
    writes.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.back().dstSet = descriptorSet;
    writes.back().dstBinding = static_cast<uint32_t>(writes.size() - 1);
    writes.back().dstArrayElement = 0;
    writes.back().descriptorType = descriptorType;
    writes.back().descriptorCount = 1;
    writes.back().pImageInfo = &imageInfo;
}

void descriptorSet::write(Writes& writes, VkDescriptorSet descriptorSet, const std::vector<VkDescriptorBufferInfo>& bufferInfos, VkDescriptorType descriptorType) {
    writes.push_back(VkWriteDescriptorSet{});
    writes.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.back().dstSet = descriptorSet;
    writes.back().dstBinding = static_cast<uint32_t>(writes.size() - 1);
    writes.back().dstArrayElement = 0;
    writes.back().descriptorType = descriptorType;
    writes.back().descriptorCount = static_cast<uint32_t>(bufferInfos.size());
    writes.back().pBufferInfo = bufferInfos.data();
}

void descriptorSet::write(Writes& writes, VkDescriptorSet descriptorSet, const std::vector<VkDescriptorImageInfo>& imageInfos, VkDescriptorType descriptorType) {
    writes.push_back(VkWriteDescriptorSet{});
    writes.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes.back().dstSet = descriptorSet;
    writes.back().dstBinding = static_cast<uint32_t>(writes.size() - 1);
    writes.back().dstArrayElement = 0;
    writes.back().descriptorType = descriptorType;
    writes.back().descriptorCount = static_cast<uint32_t>(imageInfos.size());
    writes.back().pImageInfo = imageInfos.data();
}

}
