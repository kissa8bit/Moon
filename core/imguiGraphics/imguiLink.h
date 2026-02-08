#ifndef MOON_IMGUI_GRAPHICS_IMGUILINK_H
#define MOON_IMGUI_GRAPHICS_IMGUILINK_H

#include <vulkan.h>

namespace moon::imguiGraphics {

class ImguiLink {
public:
    ImguiLink() = default;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const;
};

} // moon::imguiGraphics

#endif // MOON_IMGUI_GRAPHICS_IMGUILINK_H
