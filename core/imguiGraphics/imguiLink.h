#ifndef MOON_IMGUI_GRAPHICS_IMGUILINK_H
#define MOON_IMGUI_GRAPHICS_IMGUILINK_H

#include <graphicsManager/linkable.h>

namespace moon::imguiGraphics {

class ImguiLink : public graphicsManager::Linkable {
public:
    ImguiLink() = default;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

} // moon::imguiGraphics

#endif // MOON_IMGUI_GRAPHICS_IMGUILINK_H
