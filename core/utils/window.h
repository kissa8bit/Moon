#ifndef MOON_UTILS_WINDOW_H
#define MOON_UTILS_WINDOW_H

#include <vulkan.h>
#include <vector>

namespace moon::utils {

class Window {
public:
	struct FramebufferSize {
		uint32_t width{};
		uint32_t height{};
	};

	virtual ~Window() = default;
	virtual FramebufferSize getFramebufferSize() const = 0;
	virtual VkResult createSurface(VkInstance instance, VkSurfaceKHR* surface) const = 0;
	virtual std::vector<const char*> getExtensions() const = 0;
};

}

#endif // MOON_UTILS_WINDOW_H