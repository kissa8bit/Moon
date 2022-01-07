#ifndef MOON_INTERFACES_CAMERA_H
#define MOON_INTERFACES_CAMERA_H

#include <vulkan.h>

#include <utils/buffer.h>
#include <utils/device.h>

#include <math/linearAlgebra.h>

#include "masksGeneration.h"

namespace moon::interfaces {

#define CameraType_Value				\
	enum Value : uint32_t {		        \
        base = 1ul << 0,				\
	};
FLAG_GENERATOR(CameraType, CameraType_Value)
#undef CameraType_Value

#define CameraProperty_Value			\
	enum Value : uint32_t {			    \
		non = 0x0,					    \
	};
FLAG_GENERATOR(CameraProperty, CameraProperty_Value)
#undef CameraProperty_Value

MASK_GENERATOR(CameraMask, CameraType, CameraProperty)

class Camera{
protected:
    CameraMask mask{};

public:
    virtual ~Camera(){};

    CameraMask& lightMask() { return mask; }
    virtual utils::Buffers& buffers() = 0;
    virtual void create(const utils::PhysicalDevice& device, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;

    virtual void setTransformation(const math::mat4& transformation) = 0;
};

} // moon::interfaces

#endif // MOON_INTERFACES_CAMERA_H
