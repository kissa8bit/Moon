#ifndef MOON_UTILS_CURSOR_H
#define MOON_UTILS_CURSOR_H

#include <vulkan.h>

#include "vkdefault.h"
#include "buffer.h"

namespace moon::utils {

struct CursorPose {
    alignas(4) float x;
    alignas(4) float y;
};

struct CursorInfo {
    alignas(4) uint32_t number;
    alignas(4) float depth;
};

struct CursorBuffer {
    CursorPose pose;
    CursorInfo info;
};

class Cursor {
private:
    utils::Buffer buffer;
    CursorBuffer cursorBuffer;

public:
    void create(VkPhysicalDevice physicalDevice, VkDevice device);
    void update(const float& x, const float& y);
    const CursorBuffer& read();
    operator const CursorBuffer&() const;

    VkDescriptorBufferInfo descriptorBufferInfo() const;
};

}
#endif // MOON_UTILS_CURSOR_H
