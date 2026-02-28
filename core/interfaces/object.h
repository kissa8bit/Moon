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

#define ObjectType_Value							\
	enum Value : uint32_t {							\
		non = 0x0,									\
		baseSimple = 1ul << 0,						\
		basePBR = 1ul << 1,							\
		base = 1ul << 2,							\
		baseTypes = baseSimple | basePBR | base,	\
		skybox = 1ul << 3,							\
		outlining = 1ul << 4,						\
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

inline ObjectType objectTypeFromVertexType(Model::VertexType type)
{
	switch (type)
	{
	case Model::VertexType::baseSimple:
		return ObjectType::baseSimple;
	case Model::VertexType::basePBR:
		return ObjectType::basePBR;
	case Model::VertexType::base:
		return ObjectType::base;
	}
	return ObjectType::non;
}

inline VkVertexInputBindingDescription VertexInputBindingDescriptionFromObjectType(ObjectType type)
{
	if (type.has(ObjectType::baseSimple))
	{
		return SimpleVertex::getBindingDescription();
	}
	if (type.has(ObjectType::basePBR))
	{
		return PBRVertex::getBindingDescription();
	}
	if (type.has(ObjectType::base))
	{
		return Vertex::getBindingDescription();
	}
	return VkVertexInputBindingDescription{};
}

inline std::vector<VkVertexInputAttributeDescription> AttributeDescriptionsFromObjectType(ObjectType type)
{
	if (type.has(ObjectType::baseSimple))
	{
		return SimpleVertex::getAttributeDescriptions();
	}
	if (type.has(ObjectType::basePBR))
	{
		return PBRVertex::getAttributeDescriptions();
	}
	if (type.has(ObjectType::base))
	{
		return Vertex::getAttributeDescriptions();
	}
	return {};
}

class Object {
protected:
	ObjectMask mask{};

	Model* pModel{ nullptr };
	struct { Range range{}; } primitive;
	struct { Range range{}; } instance;

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
	const ObjectMask& objectMask() const;
	Range& primitiveRange();
	bool comparePrimitive(uint32_t primitiveIndex) const;
	const VkDescriptorSet& getDescriptorSet(uint32_t i) const;

	virtual utils::Buffers& buffers() = 0;
	virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
	virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;

	virtual void setTransformation(const math::mat4& transformation) = 0;
};

using Objects = std::vector<interfaces::Object*>;

} // moon::interfaces

#endif // MOON_INTERFACES_OBJECT_H
