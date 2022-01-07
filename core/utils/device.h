#ifndef MOON_UTILS_DEVICE_H
#define MOON_UTILS_DEVICE_H

#include <vulkan.h>
#include <string>
#include <vector>
#include <map>

#include "vkdefault.h"

namespace moon::utils {

class Device;
using Devices = std::vector<Device>;

class PhysicalDevice{
public:
    using Index = uint32_t;
    using Map = std::map<Index, PhysicalDevice>;

    struct Properties {
        Index index{ 0x7FFFFFFF };
        VkPhysicalDeviceType type{ VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM };
        VkPhysicalDeviceFeatures deviceFeatures{};
        std::string name{};
        std::vector<std::string> deviceExtensions;
        std::vector<std::string> validationLayers = { "VK_LAYER_KHRONOS_validation" };
    };

private:
    VkPhysicalDevice descriptor{ VK_NULL_HANDLE };
    Properties props{};
    QueueFamily::Map queueFamilies;
    Devices devices;

public:
    PhysicalDevice() = default;
    PhysicalDevice(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures deviceFeatures = {}, const std::vector<std::string>& deviceExtensions = {});
    PhysicalDevice& operator=(const PhysicalDevice& other) = delete;
    PhysicalDevice(const PhysicalDevice& other) = delete;
    PhysicalDevice& operator=(PhysicalDevice&& other);
    PhysicalDevice(PhysicalDevice&& other);
    void swap(PhysicalDevice& other);

    VkResult createDevice(const QueueRequest& queueRequest);
    bool presentSupport(VkSurfaceKHR surface);

    operator VkPhysicalDevice() const;
    const Device& device(uint32_t index = 0) const;
    const Properties& properties() const;
};

class Device {
private:
    VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
    VkDevice descriptor{ VK_NULL_HANDLE };
    std::map<QueueFamily::Index, std::vector<VkQueue>> queueMap;

public:
    ~Device();
    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) noexcept;
    Device& operator=(Device&& other) noexcept;
    void swap(Device& other) noexcept;

    Device(VkPhysicalDevice physicalDevice, const PhysicalDevice::Properties& properties, const QueueFamily::Map& queueFamilies, const QueueRequest& queueRequest);
    operator const VkDevice& () const;
    operator const VkDevice* () const;

    VkQueue operator()(QueueFamily::Index familyIndex, QueueIndex queueIndex) const;
};

}
#endif // MOON_UTILS_DEVICE_H
