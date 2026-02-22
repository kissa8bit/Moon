#include "cursor.h"

#include <cstring>
#include <limits>

#include "operations.h"

namespace moon::utils {

void Cursor::create(VkPhysicalDevice physicalDevice, VkDevice device) {
    buffer = utils::vkDefault::Buffer(
        physicalDevice,
        device,
        sizeof(CursorBuffer),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void Cursor::update(const float& x, const float& y) {
    CursorBuffer cursorBuffer{};
    cursorBuffer.pose.x = x;
    cursorBuffer.pose.y = y;
    cursorBuffer.info.number = std::numeric_limits<uint32_t>::max();
    buffer.copy(&cursorBuffer);
}

const CursorBuffer& Cursor::read() {
    CHECK_M(buffer.map() != nullptr, std::string("[ Cursor::read ] buffer is not host visible or not mapped"));
    std::memcpy((void*)&cursorBuffer, buffer.map(), sizeof(CursorBuffer));
    return cursorBuffer;
}

Cursor::operator const CursorBuffer& () const {
    return cursorBuffer;
}

VkDescriptorBufferInfo Cursor::descriptorBufferInfo() const {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(CursorBuffer);
    return bufferInfo;
}

}