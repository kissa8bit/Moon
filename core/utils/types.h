#ifndef MOON_UTILS_TYPES_H 
#define MOON_UTILS_TYPES_H

#include <cstdint>

namespace moon::utils {

class ImageIndex
{
private:
	uint32_t v;

public:
	explicit ImageIndex(uint32_t v) : v(v) {}

	operator uint32_t() const { return v; }
};

class ResourceIndex
{
private:
	uint32_t v;

public:
	explicit ResourceIndex(uint32_t v) : v(v) {}

	operator uint32_t() const { return v; }
};

}

#endif //  MOON_UTILS_TYPES_H 
