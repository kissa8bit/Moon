#include "buffer.h"

#include "memory.h"
#include "operations.h"

#include <cstring>

namespace moon::utils {

void createDeviceBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, const void* data, VkBufferUsageFlagBits usage, Buffer& cache, Buffer& deviceLocal) {
    cache = utils::vkDefault::Buffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    deviceLocal = utils::vkDefault::Buffer(physicalDevice, device, bufferSize, static_cast<VkBufferUsageFlags>(usage) | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Memory::instance().nameMemory(cache, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, staging");
    Memory::instance().nameMemory(deviceLocal, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, deviceLocal");

    cache.copy(data);

    buffer::copy(commandBuffer, bufferSize, cache, deviceLocal);
}

UniformBuffer::UniformBuffer(UniformBuffer&& other) noexcept {
    swap(other);
}

UniformBuffer& UniformBuffer::operator=(UniformBuffer&& other) noexcept {
    swap(other);
    return *this;
}

void UniformBuffer::swap(UniformBuffer& other) noexcept {
    std::swap(host, other.host);
    std::swap(cache, other.cache);
    std::swap(device, other.device);
    std::swap(size, other.size);
}

UniformBuffer::UniformBuffer(const void* hostData, size_t dataSize)
    : host(hostData), size(dataSize) {}

UniformBuffer::UniformBuffer(const utils::PhysicalDevice& physicalDevice, uint32_t imageCount, const void* hostData, size_t dataSize) : UniformBuffer(hostData, dataSize) {
    if (cache.empty()) {
        cache.resize(imageCount);
        for (size_t i = 0; i < cache.size(); ++i) {
            cache[i] = utils::vkDefault::Buffer(physicalDevice, physicalDevice.device(), size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            utils::Memory::instance().nameMemory(cache[i], std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", UniformBuffer::UniformBuffer, cache " + std::to_string(i));
        }
    }
    if (device.empty()) {
        device.resize(imageCount);
        for (size_t i = 0; i < device.size(); ++i) {
            device[i] = utils::vkDefault::Buffer(physicalDevice, physicalDevice.device(), size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            utils::Memory::instance().nameMemory(device[i], std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", UniformBuffer::UniformBuffer, device " + std::to_string(i));
        }
    }
}

void UniformBuffer::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    if (frameNumber >= device.size() || frameNumber >= cache.size()) return;
    if (device[frameNumber].dropFlag()) {
        cache[frameNumber].copy(host);
        utils::buffer::copy(commandBuffer, size, cache[frameNumber], device[frameNumber]);
    }
}

bool BuffersDatabase::add(const std::string& id, const Buffers* pBuffer) {
    if (!pBuffer) return false;
    buffersMap[id] = pBuffer;
    return true;
}

bool BuffersDatabase::remove(const std::string& id) {
    return buffersMap.erase(id) > 0;
}

const Buffers* BuffersDatabase::get(const std::string& id) const {
    auto it = buffersMap.find(id);
    return it != buffersMap.end() ? it->second : nullptr;
}

VkBuffer BuffersDatabase::buffer(const std::string& id, const uint32_t imageIndex) const
{
    auto it = buffersMap.find(id);
    if (it == buffersMap.end() || !(it->second) || imageIndex >= it->second->size()) return VK_NULL_HANDLE;
    return static_cast<VkBuffer>(it->second->at(imageIndex));
}

VkDescriptorBufferInfo BuffersDatabase::descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const
{
    VkDescriptorBufferInfo bufferInfo{};
    auto it = buffersMap.find(id);
    if (it == buffersMap.end() || !(it->second) || imageIndex >= it->second->size()) {
        bufferInfo.buffer = VK_NULL_HANDLE;
        bufferInfo.offset = 0;
        bufferInfo.range = 0;
        return bufferInfo;
    }

    const auto& buffer = it->second->at(imageIndex);
    bufferInfo.buffer = static_cast<VkBuffer>(buffer);
    bufferInfo.offset = 0;
    bufferInfo.range = buffer.size();
    return bufferInfo;
}

}
