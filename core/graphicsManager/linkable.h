#ifndef LINKABLE_H
#define LINKABLE_H

#include <vulkan.h>
#include "vector.h"

namespace moon::graphicsManager {

struct PositionInWindow {
    moon::math::Vector<float, 2> offset{ 0.0f, 0.0f };
    moon::math::Vector<float, 2> size{ 1.0f, 1.0f };
};

class Linkable{
protected:
    VkRenderPass pRenderPass{ VK_NULL_HANDLE };
    PositionInWindow position;

public:
    Linkable() = default;
    Linkable(VkRenderPass renderPass, const PositionInWindow& position = PositionInWindow{}) : pRenderPass(renderPass), position(position){}
    virtual ~Linkable(){};
    virtual void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const = 0;
    virtual PositionInWindow& positionInWindow() { return position; }
    virtual VkRenderPass& renderPass() { return pRenderPass; }
};

}
#endif // LINKABLE_H
