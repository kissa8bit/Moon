#ifndef MOON_IMGUI_GRAPHICS_IMGUILINK_H
#define MOON_IMGUI_GRAPHICS_IMGUILINK_H

#include <vulkan.h>

#include <utils/types.h>

namespace moon::imguiGraphics {

class ImguiLink {
public:
    ImguiLink() = default;
    void draw(VkCommandBuffer commandBuffer, utils::ResourceIndex resourceIndex) const;
};

} // moon::imguiGraphics

#endif // MOON_IMGUI_GRAPHICS_IMGUILINK_H
