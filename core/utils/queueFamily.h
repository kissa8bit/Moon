#ifndef MOON_UTILS_QUEUE_FAMILY_H
#define MOON_UTILS_QUEUE_FAMILY_H

#include <vulkan.h>
#include <vector>
#include <unordered_map>

namespace moon::utils {

using QueueIndex = uint32_t;
using QueueCount = uint32_t;

struct QueueFamily {
    using Index = uint32_t;
    using Map = std::unordered_map<Index, QueueFamily>;

    VkQueueFamilyProperties queueFamilyProperties{};
    VkBool32 presentSupport{ false };
    std::vector<float> queuePriorities;

    QueueFamily();
    QueueFamily(VkQueueFamilyProperties queueFamilyProperties, VkBool32 presentSupport = false);
    QueueFamily(const QueueFamily& other);
    QueueFamily& operator=(const QueueFamily& other);

    QueueCount count() const;
    bool availableQueueFlag(VkQueueFlags flag) const;
};

using QueueRequest = std::unordered_map<QueueFamily::Index, QueueCount>;

}

#endif // MOON_UTILS_QUEUE_FAMILY_H