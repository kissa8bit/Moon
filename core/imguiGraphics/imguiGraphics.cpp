#include "imguiGraphics.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <utils/operations.h>

namespace moon::imguiGraphics {

ImguiGraphics::ImguiGraphics(VkInstance instance, uint32_t imageCount)
    : instance(instance), imageCount(imageCount)
{
    setupImguiContext();
    link = std::make_unique<ImguiLink>();
}

ImguiGraphics::~ImguiGraphics(){
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
}

void ImguiGraphics::setupImguiContext(){
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
}

void ImguiGraphics::reset() {
    std::vector<VkDescriptorPoolSize> descriptorPoolSize = { VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSize.size());
        poolInfo.pPoolSizes = descriptorPoolSize.data();
    descriptorPool = utils::vkDefault::DescriptorPool(device->device(), poolInfo);

    ImGui_ImplVulkan_InitInfo initInfo = {};
        initInfo.Instance = instance;
        initInfo.PhysicalDevice = *device;
        initInfo.Device = device->device();
        initInfo.QueueFamily = 0;
        initInfo.Queue = device->device()(0,0);
        initInfo.PipelineCache = VK_NULL_HANDLE;
        initInfo.DescriptorPool = descriptorPool;
        initInfo.Subpass = 0;
        initInfo.MinImageCount = imageCount;
        initInfo.ImageCount = imageCount;
        initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.Allocator = VK_NULL_HANDLE;
        initInfo.CheckVkResultFn = [](VkResult result){moon::utils::debug::checkResult(result,"");};
        initInfo.RenderPass = link->renderPass();
    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();
}

void ImguiGraphics::update(uint32_t) {}

utils::vkDefault::VkSemaphores ImguiGraphics::submit(uint32_t, const utils::vkDefault::VkSemaphores& externalSemaphore) {
    return externalSemaphore;
}

} // moon::imguiGraphics
