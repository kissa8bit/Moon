#ifndef MOON_INTERFACES_LIGHT_H
#define MOON_INTERFACES_LIGHT_H

#include <vector>
#include <unordered_map>

#include <vulkan.h>

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/depthMap.h>

#include <math/linearAlgebra.h>

#include "masksGeneration.h"

namespace moon::interfaces {

#define LightType_Value		            \
	enum Value : uint32_t {		        \
        spotCircle = 1ul << 0,          \
        spotSquare = 1ul << 1,          \
	};
FLAG_GENERATOR(LightType, LightType_Value)
#undef LightType_Value

#define LightProperty_Value		        \
	enum Value : uint32_t {			    \
		non = 0x0,					    \
		enableShadow = 1ul << 0,		\
		enableScattering = 1ul << 1,	\
	};
FLAG_GENERATOR(LightProperty, LightProperty_Value)
#undef LightProperty_Value

MASK_GENERATOR(LightMask, LightType, LightProperty)

class Light
{
protected:
    LightMask mask{};

    utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::DescriptorSets descriptorSets;

public:
    Light(LightMask lightMask);

    virtual ~Light() = default;

    LightMask& lightMask();
    const VkDescriptorSet& getDescriptorSet(uint32_t i) const;

    virtual utils::Buffers& buffers() = 0;
    virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;
    virtual void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) = 0;

    virtual void setTransformation(const math::mat4& transformation) = 0;

    static utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
};

using Lights = std::vector<interfaces::Light*>;
using DepthMaps = std::unordered_map<interfaces::Light*, utils::DepthMap>;

} // moon::interfaces

#endif // MOON_INTERFACES_LIGHT_H
