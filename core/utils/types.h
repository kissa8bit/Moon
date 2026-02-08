#ifndef MOON_UTILS_TYPES_H 
#define MOON_UTILS_TYPES_H

#include <cstdint>

namespace moon::utils {

template<typename Tag>
class StrongIndex {
private:
    uint32_t value;

public:
    constexpr explicit StrongIndex(uint32_t val = 0) : value(val) {}

    constexpr uint32_t get() const { return value; }

    constexpr bool operator==(const StrongIndex& other) const { return value == other.value; }
    constexpr bool operator!=(const StrongIndex& other) const { return value != other.value; }
    constexpr bool operator<(const StrongIndex& other) const { return value < other.value; }
    constexpr bool operator<=(const StrongIndex& other) const { return value <= other.value; }
    constexpr bool operator>(const StrongIndex& other) const { return value > other.value; }
    constexpr bool operator>=(const StrongIndex& other) const { return value >= other.value; }

    constexpr StrongIndex& operator++() { ++value; return *this; }
    constexpr StrongIndex operator++(int) { StrongIndex tmp(*this); ++value; return tmp; }
    constexpr StrongIndex& operator--() { --value; return *this; }
    constexpr StrongIndex operator--(int) { StrongIndex tmp(*this); --value; return tmp; }

    constexpr StrongIndex& operator+=(uint32_t offset) { value += offset; return *this; }
    constexpr StrongIndex& operator-=(uint32_t offset) { value -= offset; return *this; }

    constexpr StrongIndex operator+(uint32_t offset) const { return StrongIndex(value + offset); }
    constexpr StrongIndex operator-(uint32_t offset) const { return StrongIndex(value - offset); }

    constexpr explicit operator uint32_t() const { return value; }
};

struct ImageIndexTag {};
struct ResourceIndexTag {};

using ImageIndex = StrongIndex<ImageIndexTag>;
using ResourceIndex = StrongIndex<ResourceIndexTag>;

template<typename Tag>
class Name {
private:
    std::string value;

public:
    explicit Name(std::string_view name = "") : value(name) {}

    const std::string& get() const { return value; }
    std::string& get() { return value; }

    operator std::string_view() const { return value; }
    operator const std::string& () const { return value; }

    bool operator==(const Name& other) const { return value == other.value; }
    bool operator!=(const Name& other) const { return value != other.value; }

	bool empty() const { return value.empty(); }
	void clear() { value.clear(); }

    Name& operator+=(const std::string& suffix) {
        value += suffix;
        return *this;
	}

    Name operator+(const std::string& suffix) const {
        return Name(value + suffix);
	}

    Name& operator+=(const Name& other) {
        value += other.value;
        return *this;
    }

    Name operator+(const Name& other) const {
        return Name(value + other.value);
	}
};

struct BufferNameTag {};
struct ImageNameTag {};
struct AttachmentNameTag {};

using BufferName = Name<BufferNameTag>;
using ImageName = Name<ImageNameTag>;
using AttachmentName = Name<AttachmentNameTag>;

template <typename T>
class DirtyValue {
private:
    T value;
    mutable bool dirty{false};

public:
    explicit DirtyValue(T val = T{}) : value(std::move(val)), dirty(true) {}

    DirtyValue(const DirtyValue& other) : value(other.value), dirty(true) {}
    DirtyValue(DirtyValue&& other) noexcept : value(std::move(other.value)), dirty(true) {}

    DirtyValue& operator=(const DirtyValue& other) {
        if (value != other.value) {
            value = other.value;
            dirty = true;
        }
        return *this;
    }

    DirtyValue& operator=(DirtyValue&& other) noexcept {
        if (value != other.value) {
            value = std::move(other.value);
            dirty = true;
        }
        return *this;
    }

    DirtyValue& operator=(const T& val) {
        if (value != val) {
            value = val;
            dirty = true;
        }
        return *this;
    }

    DirtyValue& operator=(T&& val) {
        if (value != val) {
            value = std::move(val);
            dirty = true;
        }
        return *this;
    }

    const T& get() const { return value; }

    void set(const T& val) {
        if (value != val) {
            value = val;
            dirty = true;
        }
    }

    void set(T&& val) {
        if (value != val) {
            value = std::move(val);
            dirty = true;
        }
    }

    bool isDirty() const { return dirty; }
    T consume() const { dirty = false; return value; }
    void clearDirty() const { dirty = false; }
    void markDirty() const { dirty = true; }

    explicit operator T() const { return value; }
    operator const T&() const { return value; }
};

} // moon::utils

namespace std {
    template<typename Tag>
    struct hash<moon::utils::Name<Tag>> {
        size_t operator()(const moon::utils::Name<Tag>& name) const {
            return hash<std::string>()(name.get());
        }
    };

    template<typename Tag>
    struct hash<moon::utils::StrongIndex<Tag>> {
        size_t operator()(const moon::utils::StrongIndex<Tag>& index) const {
            return hash<uint32_t>()(index.get());
        }
    };
}

#endif //  MOON_UTILS_TYPES_H 
