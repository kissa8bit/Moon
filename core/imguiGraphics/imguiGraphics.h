#ifndef MOON_IMGUI_GRAPHICS_IMGUIGRAPHICS_H
#define MOON_IMGUI_GRAPHICS_IMGUIGRAPHICS_H

#include <vector>

#include <vulkan.h>

#include <utils/vkdefault.h>

#include <graphicsManager/graphicsInterface.h>

#include "imguiLink.h"

namespace moon::imguiGraphics {

class ImguiGraphics: public graphicsManager::GraphicsInterface {
private:
    VkInstance instance{VK_NULL_HANDLE};
    uint32_t imageCount{0};

    utils::vkDefault::DescriptorPool descriptorPool;

    void setupImguiContext();

    void update(uint32_t imageIndex) override;
    utils::vkDefault::VkSemaphores submit(uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;

public:
    ImguiGraphics(VkInstance instance, uint32_t maxImageCount);
    ~ImguiGraphics();

    void reset() override;
};

} // moon::imguiGraphics

#endif // MOON_IMGUI_GRAPHICS_IMGUIGRAPHICS_H
