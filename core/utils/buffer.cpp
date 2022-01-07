#include "buffer.h"

#include "memory.h"
#include "operations.h"

#include <cstring>

namespace moon::utils {

void createDeviceBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer, size_t bufferSize, void* data, VkBufferUsageFlagBits usage, Buffer& cache, Buffer& deviceLocal) {
    cache = utils::vkDefault::Buffer(physicalDevice, device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    deviceLocal = utils::vkDefault::Buffer(physicalDevice, device, bufferSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    Memory::instance().nameMemory(cache, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, staging");
    Memory::instance().nameMemory(deviceLocal, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", createBuffer, deviceLocal");

    cache.copy(data);

    buffer::copy(commandBuffer, bufferSize, cache, deviceLocal);
};

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

UniformBuffer::UniformBuffer(void* hostData, size_t dataSize)
    : host(hostData), size(dataSize) {}

UniformBuffer::UniformBuffer(const utils::PhysicalDevice& physicalDevice, uint32_t imageCount, void* hostData, size_t dataSize) : UniformBuffer(hostData, dataSize) {
    if (cache.empty()) {
        cache.resize(imageCount);
        for (auto& buffer : cache) {
            buffer = utils::vkDefault::Buffer(physicalDevice, physicalDevice.device(), size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", UniformBuffer::UniformBuffer, cache " + std::to_string(&buffer - &cache[0]));
        }
    }
    if (device.empty()) {
        device.resize(imageCount);
        for (auto& buffer : device) {
            buffer = utils::vkDefault::Buffer(physicalDevice, physicalDevice.device(), size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            utils::Memory::instance().nameMemory(buffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", UniformBuffer::UniformBuffer, device " + std::to_string(&buffer - &device[0]));
        }
    }
}

void UniformBuffer::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    if (device[frameNumber].dropFlag()) {
        cache[frameNumber].copy(host);
        utils::buffer::copy(commandBuffer, size, cache[frameNumber], device[frameNumber]);
    }
}

bool BuffersDatabase::add(const std::string& id, const Buffers* pBuffer) {
    return buffersMap[id] = pBuffer;
}

bool BuffersDatabase::remove(const std::string& id) {
    return buffersMap.erase(id);
}

const Buffers* BuffersDatabase::get(const std::string& id) const {
    return buffersMap.find(id) != buffersMap.end() ? buffersMap.at(id) : nullptr;
}

VkBuffer BuffersDatabase::buffer(const std::string& id, const uint32_t imageIndex) const
{
    return buffersMap.find(id) != buffersMap.end() && buffersMap.at(id) ? (VkBuffer) buffersMap.at(id)->at(imageIndex) : VK_NULL_HANDLE;
}

VkDescriptorBufferInfo BuffersDatabase::descriptorBufferInfo(const std::string& id, const uint32_t imageIndex) const
{
    const auto& buffer = buffersMap.at(id)->at(imageIndex);
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = buffer.size();
    return bufferInfo;
}

}
