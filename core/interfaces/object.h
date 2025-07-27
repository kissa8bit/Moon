#ifndef OBJECT_H
#define OBJECT_H

#include <vulkan.h>
#include <vector>
#include <vector.h>

#include "vkdefault.h"
#include "device.h"
#include "buffer.h"
#include "model.h"

namespace moon::interfaces {

	class ObjectType {
	public:
		using Type = uint32_t;
		enum Value : Type {
			non = 0x0,
			base = 1ul << 0,
			skybox = 1ul << 1,
		};

		ObjectType() = default;
		ObjectType(Value value) : value(value) {}
		ObjectType(Type value) : value(static_cast<Value>(value)) {}

		operator Value() const { return value; }
		explicit operator bool() const = delete;

		bool operator==(ObjectType r) const { return value == r.value; }
		bool operator!=(ObjectType r) const { return value != r.value; }

		bool has(Value type) const { return (value & type) == type; }
		ObjectType& set(Value type, bool enable) { value = static_cast<Value>((value & ~type) | (enable ? type : 0ul)); return *this; }

	private:
		Value value{};
	};

	class ObjectProperty {
	public:
		using Type = uint32_t;
		enum Value : Type {
			non = 0x0,
			outlining = 1ul << 0,
		};

		ObjectProperty() = default;
		ObjectProperty(Value value) : value(value) {}
		ObjectProperty(Type value) : value(static_cast<Value>(value)) {}

		operator Value() const { return value; }
		explicit operator bool() const = delete;

		bool operator==(ObjectProperty r) const { return value == r.value; }
		bool operator!=(ObjectProperty r) const { return value != r.value; }

		bool has(Value prop) const { return (value & prop) == prop; }
		ObjectProperty& set(Value type, bool enable) { value = static_cast<Value>((value & ~type) | (enable ? type : 0ul)); return *this; }

	private:
		Value value{};
	};

	class ObjectMask {
	public:
		using Type = uint64_t;

		struct Hasher
		{
			Type operator()(const ObjectMask& mask) const {
				return static_cast<Type>(mask);
			}
		};

		ObjectMask(ObjectType type = ObjectType::non, ObjectProperty prop = ObjectProperty::non)
			: mask((Type)type | (Type)prop << 32)
		{}

		operator Type() const {
			return mask;
		}

		ObjectType type() const {
			return static_cast<ObjectType>(mask & 0x00000000FFFFFFFF);
		}

		ObjectProperty property() const {
			return static_cast<ObjectProperty>((mask >> 32) & 0x00000000FFFFFFFF);
		}

		ObjectMask& set(ObjectType::Value objectType, bool enable) {
			return *this = ObjectMask(type().set(objectType, enable), property());
		}

		ObjectMask& set(ObjectProperty::Value objectProperty, bool enable) {
			return *this = ObjectMask(type(), property().set(objectProperty, enable));
		}

	private:
		Type mask{ 0 };
	};

	class Object {
	protected:
		bool enable{ true };
		bool enableShadow{ true };

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

		void setEnable(const bool enable);
		void setEnableShadow(const bool enable);
		bool getEnable() const;
		bool getEnableShadow() const;

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

}
#endif // OBJECT_H
