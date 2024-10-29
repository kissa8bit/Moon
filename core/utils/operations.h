#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>

struct GLFWwindow;

namespace moon::utils {

struct ImageInfo {
    uint32_t                Count{ 0 };
    VkFormat                Format{ VK_FORMAT_UNDEFINED };
    VkExtent2D              Extent{ 0, 0 };
    VkSampleCountFlagBits   Samples{ VK_SAMPLE_COUNT_1_BIT };
};

using Paths = std::vector<std::filesystem::path>;

struct SubpassInfo {
    std::vector<VkAttachmentReference> out;
    std::vector<VkAttachmentReference> in;
    std::vector<VkAttachmentReference> depth;
};

using SubpassInfos = std::vector<SubpassInfo>;

namespace debug {

    VkResult checkResult(
            VkResult        result,
            std::string     message);

    bool checkResult(
            bool            result,
            std::string     message);

    void displayError(
            std::string     message);
}

#define CHECK(res) moon::utils::debug::checkResult(res, "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__))
#define CHECK_M(res, message) moon::utils::debug::checkResult(res, "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__) + " " + message)

namespace validationLayer{

    bool checkSupport(
            const std::vector<std::string>              validationLayers);

    void setupDebugMessenger (
            VkInstance                                  instance,
            VkDebugUtilsMessengerEXT*                   debugMessenger);

    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT             messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void*                                       pUserData);

    void destroyDebugUtilsMessengerEXT(
            VkInstance                                  instance,
            VkDebugUtilsMessengerEXT                    debugMessenger,
            const VkAllocationCallbacks*                pAllocator);
}

namespace physicalDevice {

    void printMemoryProperties(
            VkPhysicalDeviceMemoryProperties memoryProperties);

    uint32_t findMemoryTypeIndex(
            VkPhysicalDevice                physicalDevice,
            uint32_t                        memoryTypeBits,
            VkMemoryPropertyFlags           properties);

    void printQueueIndices(
            VkPhysicalDevice                device,
            VkSurfaceKHR                    surface);

    std::vector<uint32_t> findQueueFamilies(
            VkPhysicalDevice                device,
            VkSurfaceKHR                    surface = VK_NULL_HANDLE);

    std::vector<uint32_t> findQueueFamilies(
            VkPhysicalDevice                device,
            VkQueueFlagBits                 queueFlags,
            VkSurfaceKHR                    surface = VK_NULL_HANDLE);

    std::vector<VkQueueFamilyProperties> findQueueFamiliesProperties(
            VkPhysicalDevice                device,
            VkQueueFlagBits                 queueFlags,
            VkSurfaceKHR                    surface = VK_NULL_HANDLE);

    VkSampleCountFlagBits queryingSampleCount(
            VkPhysicalDevice                device);

    bool isSuitable(
            VkPhysicalDevice                device,
            VkSurfaceKHR                    surface,
            const std::vector<std::string>& deviceExtensions);

    bool isExtensionsSupport(
            VkPhysicalDevice                device,
            const std::vector<std::string>& deviceExtensions);
}

namespace buffer{

    VkResult create(
            VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkDeviceSize                    size,
            VkBufferUsageFlags              usage,
            VkMemoryPropertyFlags           properties,
            VkBuffer*                       buffer,
            VkDeviceMemory*                 bufferMemory);

    void copy(
            VkCommandBuffer                 commandBuffer,
            VkDeviceSize                    size,
            VkBuffer                        srcBuffer,
            VkBuffer                        dstBuffer);

    void destroy(
            VkDevice                        device,
            VkBuffer&                       buffer,
            VkDeviceMemory&                 memory);
}

namespace texture {

    void transitionLayout(
            VkCommandBuffer                 commandBuffer,
            VkImage                         image,
            VkImageLayout                   oldLayout,
            VkImageLayout                   newLayout,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        arrayLayers);

    void copy(
            VkCommandBuffer                 commandBuffer,
            VkBuffer                        srcBuffer,
            VkImage                         dstImage,
            VkExtent3D                      extent,
            uint32_t                        layerCount);

    void copy(
            VkCommandBuffer                 commandBuffer,
            VkImage                         srcImage,
            VkBuffer                        dstBuffer,
            VkExtent3D                      extent,
            uint32_t                        layerCount);

    VkResult create(
            VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkImageCreateFlags              flags,
            VkExtent3D                      extent,
            uint32_t                        arrayLayers,
            uint32_t                        mipLevels,
            VkSampleCountFlagBits           numSamples,
            VkFormat                        format,
            VkImageLayout                   layout,
            VkImageUsageFlags               usage,
            VkMemoryPropertyFlags           properties,
            VkImage*                        image,
            VkDeviceMemory*                 imageMemory);

    void destroy(
            VkDevice                            device,
            VkImage&                            image,
            VkDeviceMemory&                     memory);

    VkResult createView(
            VkDevice                        device,
            VkImageViewType                 type,
            VkFormat                        format,
            VkImageAspectFlags              aspectFlags,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            VkImage                         image,
            VkImageView*                    imageView);

    void generateMipmaps(
            VkPhysicalDevice                physicalDevice,
            VkCommandBuffer                 commandBuffer,
            VkImage                         image,
            VkFormat                        imageFormat,
            int32_t                         texWidth,
            int32_t                         texHeight,
            uint32_t                        mipLevels,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount);

    void blitDown(
            VkCommandBuffer                 commandBuffer,
            VkImage                         srcImage,
            uint32_t                        srcMipLevel,
            VkImage                         dstImage,
            uint32_t                        dstMipLevel,
            uint32_t                        width,
            uint32_t                        height,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            float                           blitFactor);

    void blitUp(
            VkCommandBuffer                 commandBuffer,
            VkImage                         srcImage,
            uint32_t                        srcMipLevel,
            VkImage                         dstImage,
            uint32_t                        dstMipLevel,
            uint32_t                        width,
            uint32_t                        height,
            uint32_t                        baseArrayLayer,
            uint32_t                        layerCount,
            float                           blitFactor);
}

namespace singleCommandBuffer {

    VkCommandBuffer create(
            VkDevice                        device,
            VkCommandPool                   commandPool);

    VkResult submit(
            VkDevice                        device,
            VkQueue                         queue,
            VkCommandPool                   commandPool,
            VkCommandBuffer*                commandBuffer);

    VkResult submit(
            VkDevice                        device,
            VkQueue                         queue,
            VkCommandPool                   commandPool,
            uint32_t                        commandBufferCount,
            VkCommandBuffer*                commandBuffer);

    class Scoped{
    private:
        VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
        VkDevice device{VK_NULL_HANDLE};
        VkQueue queue{VK_NULL_HANDLE};
        VkCommandPool commandPool{VK_NULL_HANDLE};
    public:
        Scoped(VkDevice device, VkQueue queue, VkCommandPool commandPool);
        ~Scoped();
        Scoped(const Scoped&) = delete;
        Scoped& operator=(const Scoped&) = delete;
        operator VkCommandBuffer();
    };
}

template<typename T>
void raiseFlags(std::vector<T>& buffers) {
    for (auto& buffer : buffers) buffer.raiseFlag();
}

namespace swapChain {

    struct SupportDetails{
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
        bool isNotEmpty(){return !formats.empty() && !presentModes.empty();}
    };

    SupportDetails queryingSupport(
            VkPhysicalDevice                        device,
            VkSurfaceKHR                            surface);

    uint32_t queryingSupportImageCount(
            VkPhysicalDevice                        device,
            VkSurfaceKHR                            surface);

    VkSurfaceFormatKHR queryingSurfaceFormat(
            const std::vector<VkSurfaceFormatKHR>&  availableFormats);

    VkPresentModeKHR queryingPresentMode(
            const std::vector<VkPresentModeKHR>&    availablePresentModes);

    VkExtent2D queryingExtent(
            GLFWwindow* window,
            const VkSurfaceCapabilitiesKHR&         capabilities);
}

namespace image{

    VkFormat supportedFormat(
            VkPhysicalDevice                physicalDevice,
            const std::vector<VkFormat>&    candidates,
            VkImageTiling                   tiling,
            VkFormatFeatureFlags            features);

    std::vector<VkFormat> depthFormats();

    VkFormat depthStencilFormat(
            VkPhysicalDevice                physicalDevice);
}

namespace shaderModule {

    std::vector<char> readFile(
            const std::filesystem::path&    filename);

    VkShaderModule create(
            VkDevice                        device,
            const std::vector<char>&        code);
}

}
#endif
