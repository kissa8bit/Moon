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

    ImguiLink linkMember;

	bool isImGuilVulkanInit{ false };

    void setupImguiContext();

    void update(utils::ResourceIndex resourceIndex) override;
    utils::vkDefault::VkSemaphores submit(utils::ResourceIndex resourceIndex, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;
    void draw(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) const override;

public:
    ImguiGraphics(VkInstance instance, uint32_t maxImageCount);
    ~ImguiGraphics();

    void reset() override;
};

} // moon::imguiGraphics

#endif // MOON_IMGUI_GRAPHICS_IMGUIGRAPHICS_H
