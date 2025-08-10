#ifndef MOON_INTERFACES_MASKS_GENERATION_H
#define MOON_INTERFACES_MASKS_GENERATION_H

#include <type_traits>

#ifndef FLAG_GENERATOR
#define FLAG_GENERATOR(FlagName, FlagValueEnum)															\
	class FlagName {																					\
	public:																								\
		FlagValueEnum																					\
		using Type = std::underlying_type_t<Value>;														\
																										\
		struct Hasher {																					\
			Type operator()(const FlagName& mask) const {												\
				return static_cast<Type>(mask);															\
			}																							\
		};																								\
																										\
		FlagName() = default;																			\
		FlagName(Value value) : value(value) {}															\
		FlagName(Type value) : value(static_cast<Value>(value)) {}										\
																										\
		operator Value() const { return value; }														\
		explicit operator bool() const = delete;														\
																										\
		bool operator==(FlagName r) const { return value == r.value; }									\
		bool operator!=(FlagName r) const { return value != r.value; }									\
																										\
		bool has(Value type) const { return (value & type) == type; }									\
		bool has_any(Value type) const { return value & type; }											\
		FlagName& set(Value type, bool enable) {														\
			value = static_cast<Value>((value & ~type) | (enable ? type : 0ul));						\
			return *this;																				\
		}																								\
																										\
	private:																							\
		Value value{};																					\
	};

#endif // FLAG_GENERATOR

#ifndef MASK_GENERATOR
#define MASK_GENERATOR(MaskName, TypeFlag, PropFlag)													\
	class MaskName {																					\
	public:																								\
		using Type = uint64_t;																			\
		static_assert(std::is_same_v<std::underlying_type_t<TypeFlag::Value>, uint32_t>);				\
		static_assert(std::is_same_v<std::underlying_type_t<PropFlag::Value>, uint32_t>);				\
																										\
		struct Hasher {																					\
			Type operator()(const MaskName& mask) const {												\
				return static_cast<Type>(mask);															\
			}																							\
		};																								\
																										\
		MaskName(TypeFlag type = static_cast<TypeFlag>(0), PropFlag prop = static_cast<PropFlag>(0))	\
			: mask((Type)type | (Type)prop << 32)														\
		{}																								\
																										\
		operator Type() const {																			\
			return mask;																				\
		}																								\
																										\
		TypeFlag type() const {																			\
			return static_cast<TypeFlag>(mask & 0x00000000FFFFFFFF);									\
		}																								\
																										\
		PropFlag property() const {																		\
			return static_cast<PropFlag>((mask >> 32) & 0x00000000FFFFFFFF);							\
		}																								\
																										\
		MaskName& set(TypeFlag::Value objectType, bool enable) {										\
			return *this = MaskName(type().set(objectType, enable), property());						\
		}																								\
																										\
		MaskName& set(PropFlag::Value objectProperty, bool enable) {									\
			return *this = MaskName(type(), property().set(objectProperty, enable));					\
		}																								\
																										\
	private:																							\
		Type mask{ 0 };																					\
	};

#endif // MASK_GENERATOR

#endif // MOON_INTERFACES_MASKS_GENERATION_H
