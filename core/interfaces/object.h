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
		simple = 1ul << 0,							\
		pbr = 1ul << 1,								\
		animated = 1ul << 2,						\
		baseTypes = simple | pbr | animated,		\
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
	case Model::VertexType::simple:
		return ObjectType::simple;
	case Model::VertexType::pbr:
		return ObjectType::pbr;
	case Model::VertexType::animated:
		return ObjectType::animated;
	}
	return ObjectType::non;
}

inline VkVertexInputBindingDescription VertexInputBindingDescriptionFromObjectType(ObjectType type)
{
	if (type.has(ObjectType::simple))
	{
		return SimpleVertex::getBindingDescription();
	}
	if (type.has(ObjectType::pbr))
	{
		return PBRVertex::getBindingDescription();
	}
	if (type.has(ObjectType::animated))
	{
		return Vertex::getBindingDescription();
	}
	return VkVertexInputBindingDescription{};
}

inline std::vector<VkVertexInputAttributeDescription> AttributeDescriptionsFromObjectType(ObjectType type)
{
	if (type.has(ObjectType::simple))
	{
		return SimpleVertex::getAttributeDescriptions();
	}
	if (type.has(ObjectType::pbr))
	{
		return PBRVertex::getAttributeDescriptions();
	}
	if (type.has(ObjectType::animated))
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
	const Model* model() const;
	uint32_t getInstanceNumber(utils::ResourceIndex resourceIndex) const;
	uint32_t instanceCount() const;

	ObjectMask& objectMask();
	const ObjectMask& objectMask() const;
	Range& primitiveRange();
	bool comparePrimitive(uint32_t primitiveIndex) const;
	const VkDescriptorSet& getDescriptorSet(utils::ResourceIndex resourceIndex) const;

	virtual utils::Buffers& buffers() = 0;
	virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
	virtual void update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) = 0;

	virtual void setTransformation(const math::mat4& transformation) = 0;
};

using Objects = std::vector<interfaces::Object*>;

} // moon::interfaces

#endif // MOON_INTERFACES_OBJECT_H
