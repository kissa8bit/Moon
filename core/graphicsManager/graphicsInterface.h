#ifndef MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H
#define MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H

#include <vector>
#include <memory>

#include <vulkan.h>

#include <utils/device.h>
#include <utils/swapChain.h>

#include <math/linearAlgebra.h>

namespace moon::graphicsManager {

struct PositionInWindow {
    math::vec2 offset{ 0.0f, 0.0f };
    math::vec2 size{ 1.0f, 1.0f };
};

class GraphicsInterface{
protected:
    const utils::PhysicalDevice::Map* devices{ nullptr };
    const utils::PhysicalDevice* device{ nullptr };
    const utils::SwapChain* swapChainKHR{ nullptr };

    uint32_t resourceCount{ 0 };

    VkRenderPass pRenderPass{ VK_NULL_HANDLE };
    PositionInWindow position;

private:
    virtual void update(uint32_t imageIndex) = 0;
    virtual utils::vkDefault::VkSemaphores submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore) = 0;
    virtual void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const = 0;

    virtual void setProperties(
        const utils::PhysicalDevice::Map& devicesMap,
        const utils::PhysicalDevice::Index deviceIndex,
        const utils::SwapChain* swapChain,
        uint32_t resources,
        VkRenderPass renderPass)
    {
        pRenderPass = renderPass;
        swapChainKHR = swapChain;
        resourceCount = resources;
        devices = &devicesMap;
        device = &devicesMap.at(deviceIndex);
    }

    friend class GraphicsLinker;
    friend class GraphicsManager;

public:
    virtual ~GraphicsInterface(){};

    virtual void reset() = 0;

    virtual void setPositionInWindow(const PositionInWindow& pos) {
        position = pos;
    }

    virtual PositionInWindow getPositionInWindow() const {
        return position;
    }
};

} // moon::graphicsManager

#endif // MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H
