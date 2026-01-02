#ifndef MOON_UTILS_BUFFER_H
#define MOON_UTILS_BUFFER_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

#include "vkdefault.h"
#include "device.h"

namespace moon::utils {

using Buffer = vkDefault::Buffer;
using Buffers = std::vector<Buffer>;

void createDeviceBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, const void* data, VkBufferUsageFlagBits usage, Buffer& cache, Buffer& deviceLocal);

struct UniformBuffer {
    const void* host{ nullptr };
    utils::Buffers cache;
    utils::Buffers device;
    size_t size{ 0 };

    UniformBuffer() = default;
    UniformBuffer(const void* hostData, size_t dataSize);
    UniformBuffer(const utils::PhysicalDevice& device, uint32_t imageCount, const void* hostData, size_t dataSize);
    UniformBuffer(UniformBuffer&&) noexcept;
    UniformBuffer& operator=(UniformBuffer&&) noexcept;
    void swap(UniformBuffer&) noexcept;

    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer);
};

struct BuffersDatabase{
    std::unordered_map<std::string, const Buffers*> buffersMap;

    BuffersDatabase() = default;
    BuffersDatabase(const BuffersDatabase&) = default;
    BuffersDatabase& operator=(const BuffersDatabase&) = default;

    bool add(const std::string& id, const Buffers* pBuffer);
    bool remove(const std::string& id);
    const Buffers* get(const std::string& id) const;
    VkBuffer buffer(const std::string& id, const uint32_t imageIndex) const;
    VkDescriptorBufferInfo descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const;
};

}
#endif // MOON_UTILS_BUFFER_H
