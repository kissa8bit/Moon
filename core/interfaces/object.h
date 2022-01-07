#ifndef MOON_INTERFACES_OBJECT_H
#define MOON_INTERFACES_OBJECT_H

#include <vulkan.h>
#include <vector>

#include <vector.h>

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>

#include "model.h"
#include "masksGeneration.h"

namespace moon::interfaces {

#define ObjectType_Value		\
	enum Value : uint32_t {		\
		non = 0x0,				\
		base = 1ul << 0,		\
		skybox = 1ul << 1,		\
		outlining = 1ul << 2,	\
	};
FLAG_GENERATOR(ObjectType, ObjectType_Value)
#undef ObjectType_Value

#define ObjectProperty_Value		\
	enum Value : uint32_t {			\
		non = 0x0,					\
		enable = 1ul << 0,			\
		enableShadow = 1ul << 1,	\
	};
FLAG_GENERATOR(ObjectProperty, ObjectProperty_Value)
#undef ObjectProperty_Value

MASK_GENERATOR(ObjectMask, ObjectType, ObjectProperty)

class Object {
protected:
	struct { Range range{}; } primitive;
	struct { Range range{}; } instance;

	ObjectMask mask{};
	Model* pModel{ nullptr };

	utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
	utils::vkDefault::DescriptorPool descriptorPool;
	utils::vkDefault::DescriptorSets descriptors;

public:
	Object() = default;
	Object(ObjectMask objectMask, Model* model = nullptr, const Range& instanceRange = { 0,1 });

	virtual ~Object() = default;

	Model* model();
	uint32_t getInstanceNumber(uint32_t imageNumber) const;

	ObjectMask& objectMask();
	Range& primitiveRange();
	bool comparePrimitive(uint32_t primitiveIndex) const;
	const VkDescriptorSet& getDescriptorSet(uint32_t i) const;

	virtual utils::Buffers& buffers() = 0;
	virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
	virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;

	static utils::vkDefault::DescriptorSetLayout createBaseDescriptorSetLayout(VkDevice device);
	static utils::vkDefault::DescriptorSetLayout createSkyboxDescriptorSetLayout(VkDevice device);
};

using Objects = std::vector<interfaces::Object*>;

} // moon::interfaces

#endif // MOON_INTERFACES_OBJECT_H
