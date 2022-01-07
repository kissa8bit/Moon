#ifndef MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H
#define MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H

#include <vector>
#include <memory>

#include <vulkan.h>

#include <utils/device.h>
#include <utils/swapChain.h>

#include <math/vector.h>

#include "linkable.h"

namespace moon::graphicsManager {

class GraphicsInterface{
protected:
    const utils::PhysicalDevice::Map* devices{ nullptr };
    const utils::PhysicalDevice* device{ nullptr };
    const utils::SwapChain* swapChainKHR{ nullptr };

    uint32_t resourceCount{ 0 };
    std::unique_ptr<Linkable> link;

private:
    virtual void update(uint32_t imageIndex) = 0;
    virtual utils::vkDefault::VkSemaphores submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore) = 0;

    virtual void setProperties(
        const utils::PhysicalDevice::Map& devicesMap,
        const uint32_t deviceIndex,
        const utils::SwapChain* swapChain,
        uint32_t resources)
    {
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

    virtual void setPositionInWindow(const PositionInWindow& position) {
        if(link.get()) link->positionInWindow() = position;
    }

    virtual PositionInWindow getPositionInWindow() const {
        return link.get() ? link->positionInWindow() : PositionInWindow{};
    }
};

} // moon::graphicsManager

#endif // MOON_GRAPHICS_MANAGER_GRAPHICSINTERFACE_H
