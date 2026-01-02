#include "queueFamily.h"

namespace moon::utils {

QueueFamily::QueueFamily() = default;

QueueFamily::QueueFamily(VkQueueFamilyProperties queueFamilyProperties, VkBool32 presentSupport)
    : queueFamilyProperties(queueFamilyProperties), presentSupport(presentSupport)
{
    const auto qcount = queueFamilyProperties.queueCount;
    const float priority = qcount ? (1.0f / static_cast<float>(qcount)) : 0.0f;
    queuePriorities.assign(static_cast<size_t>(qcount), priority);
}

QueueFamily::QueueFamily(const QueueFamily & other)
    : QueueFamily(other.queueFamilyProperties, other.presentSupport)
{}

QueueFamily& QueueFamily::operator=(const QueueFamily & other) {
    queueFamilyProperties = other.queueFamilyProperties;
    presentSupport = other.presentSupport;
    const auto qcount = queueFamilyProperties.queueCount;
    const float priority = qcount ? (1.0f / static_cast<float>(qcount)) : 0.0f;
    queuePriorities.assign(static_cast<size_t>(qcount), priority);
    return *this;
}

QueueCount QueueFamily::count() const {
    return queueFamilyProperties.queueCount;
}

bool QueueFamily::availableQueueFlag(VkQueueFlags flag) const {
    return (flag & queueFamilyProperties.queueFlags) == flag;
}

}