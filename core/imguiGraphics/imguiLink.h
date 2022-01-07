#ifndef IMGUILINK_H
#define IMGUILINK_H

#include "linkable.h"

namespace moon::imguiGraphics {

class ImguiLink : public moon::graphicsManager::Linkable {
public:
    ImguiLink() = default;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

}
#endif // IMGUILINK_H
