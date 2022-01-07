#include "queueFamily.h"

namespace moon::utils {

QueueFamily::QueueFamily() = default;

QueueFamily::QueueFamily(VkQueueFamilyProperties queueFamilyProperties, VkBool32 presentSupport)
    : queueFamilyProperties(queueFamilyProperties), presentSupport(presentSupport)
{
    queuePriorities.resize(queueFamilyProperties.queueCount, 1.0f / queueFamilyProperties.queueCount);
}

QueueFamily::QueueFamily(const QueueFamily & other)
    : QueueFamily(other.queueFamilyProperties, other.presentSupport)
{}

QueueFamily& QueueFamily::operator=(const QueueFamily & other) {
    queueFamilyProperties = other.queueFamilyProperties;
    presentSupport = other.presentSupport;
    queuePriorities.resize(queueFamilyProperties.queueCount, 1.0f / queueFamilyProperties.queueCount);
    return *this;
}

QueueCount QueueFamily::count() const {
    return queueFamilyProperties.queueCount;
}

bool QueueFamily::availableQueueFlag(VkQueueFlags flag) const {
    return (flag & queueFamilyProperties.queueFlags) == flag;
}

}