#include "device.h"
#include "operations.h"

namespace moon::utils {

Device::~Device() {
    if (descriptor) {
        vkDestroyDevice(descriptor, nullptr);
    }
    descriptor = VK_NULL_HANDLE;
}

Device::Device(Device&& other) noexcept {
    swap(other);
}

Device& Device::operator=(Device&& other) noexcept {
    swap(other);
    return *this;
}

void Device::swap(Device& other) noexcept {
    std::swap(physicalDevice, other.physicalDevice);
    std::swap(descriptor, other.descriptor);
    std::swap(queueMap, other.queueMap);
}

Device::Device(VkPhysicalDevice physicalDevice, const PhysicalDevice::Properties& properties, const QueueFamily::Map& queueFamilies, const QueueRequest& queueRequest)
    : physicalDevice(physicalDevice)
{
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (const auto& [index, request] : queueRequest) {
        if (const auto it = queueFamilies.find(index); it != queueFamilies.end()) {
            const auto& queueFamily = it->second;
            auto& queueCreateInfo = queueCreateInfos.emplace_back(VkDeviceQueueCreateInfo{});
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = index;
            queueCreateInfo.queueCount = std::min(queueFamily.count(), request);
            queueCreateInfo.pQueuePriorities = queueFamily.queuePriorities.data();
        }
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &properties.deviceFeatures;

    std::vector<const char*> enabledExtensionNames;
    enabledExtensionNames.reserve(properties.deviceExtensions.size());
    for (const auto& ext : properties.deviceExtensions) enabledExtensionNames.push_back(ext.c_str());
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNames.size());
    createInfo.ppEnabledExtensionNames = enabledExtensionNames.empty() ? nullptr : enabledExtensionNames.data();

#ifndef DEBUG_PRINT_DISABLE
    std::vector<const char*> enabledLayerNames;
    enabledLayerNames.reserve(properties.validationLayers.size());
    for (const auto& layer : properties.validationLayers) enabledLayerNames.push_back(layer.c_str());
    createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size());
    createInfo.ppEnabledLayerNames = enabledLayerNames.empty() ? nullptr : enabledLayerNames.data();
#endif

    CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &descriptor));

    for (const auto& queueCreateInfo : queueCreateInfos) {
        queueMap[queueCreateInfo.queueFamilyIndex] = std::vector<VkQueue>(queueCreateInfo.queueCount);
        for (uint32_t index = 0; index < queueCreateInfo.queueCount; index++) {
            vkGetDeviceQueue(descriptor, queueCreateInfo.queueFamilyIndex, index, &queueMap[queueCreateInfo.queueFamilyIndex][index]);
        }
    }
}

Device::operator const VkDevice& () const {
    return descriptor;
}

Device::operator const VkDevice* () const {
    return &descriptor;
}

VkQueue Device::operator()(QueueFamily::Index familyIndex, QueueIndex queueIndex) const {
    if (queueMap.count(familyIndex)) {
        if (const auto& queue = queueMap.at(familyIndex); queue.size() > queueIndex) {
            return queue[queueIndex];
        }
    }
    return VK_NULL_HANDLE;
}

PhysicalDevice& PhysicalDevice::operator=(PhysicalDevice&& other) {
    swap(other);
    return *this;
};

PhysicalDevice::PhysicalDevice(PhysicalDevice&& other) {
    swap(other);
};

void PhysicalDevice::swap(PhysicalDevice& other) {
    std::swap(descriptor, other.descriptor);
    std::swap(props, other.props);
    std::swap(queueFamilies, other.queueFamilies);
    std::swap(devices, other.devices);
}

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures deviceFeatures, const std::vector<std::string>& deviceExtensions)
    : descriptor(physicalDevice)
{
    uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties.data());

    for (uint32_t index = 0; index < queueFamilyPropertyCount; index++) {
        queueFamilies[index] = QueueFamily(queueFamilyProperties[index]);
    }

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    props.index = physicalDeviceProperties.deviceID;
    props.type = physicalDeviceProperties.deviceType;
    props.deviceFeatures = deviceFeatures;
    props.name = physicalDeviceProperties.deviceName;
    props.deviceExtensions = deviceExtensions;
}

bool PhysicalDevice::presentSupport(VkSurfaceKHR surface)
{
    VkBool32 presentSupport = false;
    if(surface){
        for (auto& [index, family] : queueFamilies){
            VkBool32 support = false;
            CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(descriptor, index, surface, &support));
            presentSupport |= support;
            family.presentSupport = support;
        }
    }
    return presentSupport;
}

VkResult PhysicalDevice::createDevice(const QueueRequest& queueRequest)
{
    devices.emplace_back(descriptor, props, queueFamilies, queueRequest);
    return VK_SUCCESS;
}

PhysicalDevice::operator VkPhysicalDevice() const {
    return descriptor;
}

const Device& PhysicalDevice::device(uint32_t index) const {
    return devices.at(index);
}

const PhysicalDevice::Properties& PhysicalDevice::properties() const {
    return props;
}

}
