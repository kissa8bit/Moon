#ifndef CAMERA_H
#define CAMERA_H

#include <vulkan.h>
#include "buffer.h"
#include "device.h"

namespace moon::interfaces {

class Camera{
public:
    virtual ~Camera(){};

    virtual utils::Buffers& buffers() = 0;
    virtual void create(const utils::PhysicalDevice& device, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;
};

}
#endif // CAMERA_H
