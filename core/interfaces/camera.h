#ifndef MOON_INTERFACES_CAMERA_H
#define MOON_INTERFACES_CAMERA_H

#include <vulkan.h>

#include <utils/vkdefault.h>
#include <utils/buffer.h>
#include <utils/device.h>
#include <utils/types.h>

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

    utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::DescriptorSets descriptorSets;

public:
    virtual ~Camera(){};

    CameraMask& cameraMask() { return mask; }
    const VkDescriptorSet& getDescriptorSet(utils::ResourceIndex resourceIndex) const;
    virtual utils::Buffers& buffers() = 0;
    virtual void create(const utils::PhysicalDevice& device, uint32_t imageCount) = 0;
    virtual void update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) = 0;

    virtual void setTransformation(const math::mat4& transformation) = 0;
    virtual void setViewport(float width, float height) = 0;

    static utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
};

} // moon::interfaces

#endif // MOON_INTERFACES_CAMERA_H
