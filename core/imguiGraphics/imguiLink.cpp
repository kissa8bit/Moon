#include "imguiLink.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

namespace moon::imguiGraphics {

void ImguiLink::draw(VkCommandBuffer commandBuffer, uint32_t) const {
    // Rendering
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
}

}
