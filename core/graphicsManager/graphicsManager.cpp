#include "graphicsManager.h"
#include "linkable.h"

#include <string>

namespace moon::graphicsManager {

GraphicsManager::GraphicsManager(utils::Window* window, int32_t imageCount, int32_t resourceCount, const VkPhysicalDeviceFeatures& deviceFeatures) :
    imageCount(imageCount),
    resourceCount(resourceCount) {
    moon::utils::debug::checkResult(createInstance(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createDevice(deviceFeatures), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createSurface(window), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    reset(window);
}

void GraphicsManager::reset(utils::Window* window){
    deviceWaitIdle();
    moon::utils::debug::checkResult(createSwapChain(window, imageCount), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createLinker(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
    moon::utils::debug::checkResult(createSyncObjects(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));
}

VkResult GraphicsManager::createInstance(utils::Window* window){
    VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Moon Graphics Manager";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

    std::vector<const char*> extensions;
    if(window){
        extensions = window->getExtensions();
    }

    bool enableValidationLayers = moon::utils::validationLayer::checkSupport(validationLayers);
    if(enableValidationLayers){
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = moon::utils::validationLayer::debugCallback;
    VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.enabledLayerCount = enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0;
        createInfo.ppEnabledLayerNames = enableValidationLayers ? validationLayers.data() : nullptr;
        createInfo.pNext = enableValidationLayers ? (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo : nullptr;

    instance = utils::vkDefault::Instance(createInfo);

    if (enableValidationLayers) {
        debugMessenger = utils::vkDefault::DebugUtilsMessenger(instance);
    }

    return VK_SUCCESS;
}

VkResult GraphicsManager::createDevice(const VkPhysicalDeviceFeatures& deviceFeatures){
    CHECK_M(instance, "[ GraphicsManager::createDevice ] instance is VK_NULL_HANDLE");

    VkResult result = VK_SUCCESS;

    uint32_t deviceCount = 0;
    CHECK(result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));

    std::vector<VkPhysicalDevice> phDevices(deviceCount);
    CHECK(result = vkEnumeratePhysicalDevices(instance, &deviceCount, phDevices.data()));

    for (const auto phDevice : phDevices){
        auto device = moon::utils::PhysicalDevice(phDevice, deviceFeatures, deviceExtensions);
        const moon::utils::PhysicalDevice::Index index = device.properties().index;
        devices[index] = std::move(device);
        if(!activeDevice){
            activeDevice = &devices[index];
        }
        if(activeDevice->properties().type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            devices[index].properties().type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){
            activeDevice = &devices[index];
        }
    }

    if(!activeDevice) return VK_ERROR_DEVICE_LOST;

    activeDevice->createDevice({ {0,2} });

    return VK_SUCCESS;
}

VkResult GraphicsManager::createSurface(utils::Window* window){
    CHECK_M(instance, "[ GraphicsManager::createSurface ] instance is VK_NULL_HANDLE");
    CHECK_M(!devices.empty(), "[ GraphicsManager::createSurface ] device is VK_NULL_HANDLE");
    CHECK_M(window, "[ GraphicsManager::createSurface ] Window is nullptr");

    surface = utils::vkDefault::Surface(instance, window);
    CHECK_M(activeDevice->presentSupport(surface), "[ GraphicsManager::createSurface ] device doesn't support present");

    return VK_SUCCESS;
}

VkResult GraphicsManager::createSwapChain(utils::Window* window, int32_t maxImageCount){
    CHECK_M(window, "[ GraphicsManager::createSwapChain ] Window is nullptr");
    CHECK_M(surface, "[ GraphicsManager::createSwapChain ] surface is VK_NULL_HANDLE");
    CHECK_M(activeDevice, "[ GraphicsManager::activeDevice ] device is nullptr");
    return swapChainKHR.reset(activeDevice, window, surface, maxImageCount);
}

VkResult GraphicsManager::createLinker(){
    linker = GraphicsLinker(activeDevice->device(), &swapChainKHR, &graphics);
    return VK_SUCCESS;
}

void GraphicsManager::setGraphics(GraphicsInterface* ingraphics){
    graphics.push_back(ingraphics);
    ingraphics->setProperties(devices, activeDevice->properties().index, &swapChainKHR, resourceCount);
    ingraphics->link->renderPass() = linker.getRenderPass();
}

void GraphicsManager::setDevice(utils::PhysicalDevice::Index deviceIndex){
    activeDevice = &devices.at(deviceIndex);
}

VkResult GraphicsManager::createSyncObjects(){
    CHECK_M(!devices.empty(), "[ GraphicsManager::createSyncObjects ] there are no created devices");

    VkResult result = VK_SUCCESS;

    availableSemaphores.resize(resourceCount);
    for (auto& semaphore : availableSemaphores) {
        semaphore = utils::vkDefault::Semaphore(activeDevice->device());
    }

    fences.resize(resourceCount);
    for (auto& fence : fences) {
        fence = utils::vkDefault::Fence(activeDevice->device());
    }

    return result;
}

VkResult GraphicsManager::checkNextFrame(){
#define GM_CNF_RET(expr) if (auto result = CHECK(expr); result) return result;
    GM_CNF_RET(vkWaitForFences(activeDevice->device(), 1, fences[resourceIndex], VK_TRUE, UINT64_MAX))
    GM_CNF_RET(vkResetFences(activeDevice->device(), 1, fences[resourceIndex]))
    GM_CNF_RET(swapChainKHR.acquireNextImage(availableSemaphores[resourceIndex], imageIndex))
    return VK_SUCCESS;
#undef GM_CNF_RET
}

VkResult GraphicsManager::drawFrame(){
    for(auto graph : graphics){
        graph->update(resourceIndex);
    }
    linker.update(resourceIndex, imageIndex);

    utils::vkDefault::VkSemaphores waitSemaphores = {availableSemaphores[resourceIndex]};
    for(auto graph: graphics){
        waitSemaphores = graph->submit(resourceIndex, waitSemaphores);
    }

    VkSemaphore linkerSemaphore = linker.submit(imageIndex, waitSemaphores, fences[resourceIndex], activeDevice->device()(0,0));

    resourceIndex = utils::ResourceIndex((resourceIndex + 1) % resourceCount);

    return swapChainKHR.present(linkerSemaphore, imageIndex);
}

VkResult GraphicsManager::deviceWaitIdle() const {
    return vkDeviceWaitIdle(activeDevice->device());
}

VkInstance      GraphicsManager::getInstance()      const {return instance;}
VkExtent2D      GraphicsManager::getImageExtent()   const {return swapChainKHR.info().Extent;}
utils::ResourceIndex    GraphicsManager::getResourceIndex() const {return resourceIndex;}
uint32_t                GraphicsManager::getResourceCount() const {return resourceCount;}
utils::ImageIndex       GraphicsManager::getImageIndex()    const {return imageIndex;}
uint32_t                GraphicsManager::getImageCount()    const {return imageCount;}

std::vector<uint32_t> GraphicsManager::makeScreenshot() const {
    return swapChainKHR.makeScreenshot(imageIndex);
}

} // moon::graphicsManager
