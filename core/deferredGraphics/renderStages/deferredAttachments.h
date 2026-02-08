#ifndef MOON_DEFERRED_GRAPHICS_RENDER_STAGES_DEFERREDATTACHMENTS_H
#define MOON_DEFERRED_GRAPHICS_RENDER_STAGES_DEFERREDATTACHMENTS_H

#include <utils/attachments.h>

namespace moon::deferredGraphics {

struct DeferredAttachments{
private:
    std::vector<utils::Attachments> attachments;

    class GBuffer {
    public:
        constexpr static uint32_t size() { return depthIndex() + 1; }
        constexpr static uint32_t positionIndex() { return 0; }
        constexpr static uint32_t normalIndex() { return 1; }
        constexpr static uint32_t colorIndex() { return 2; }
        constexpr static uint32_t emissionIndex() { return 3; }
        constexpr static uint32_t depthIndex() { return 4; }
    };

public:
    DeferredAttachments() { attachments.resize(size());}

    constexpr static uint32_t size() {return GBufferOffset() + GBuffer::size();}
    constexpr static uint32_t GBufferOffset() { return bloomIndex() + 1; }
    constexpr static uint32_t imageIndex() { return 0;}
    constexpr static uint32_t bloomIndex() {return 1;}
    constexpr static uint32_t positionIndex() { return GBufferOffset() + GBuffer::positionIndex(); }
    constexpr static uint32_t normalIndex() { return GBufferOffset() + GBuffer::normalIndex(); }
    constexpr static uint32_t colorIndex() { return GBufferOffset() + GBuffer::colorIndex(); }
    constexpr static uint32_t emissionIndex() { return GBufferOffset() + GBuffer::emissionIndex(); }
    constexpr static uint32_t depthIndex() { return GBufferOffset() + GBuffer::depthIndex(); }

    utils::Attachments& image() { return attachments[imageIndex()]; }
    utils::Attachments& bloom() { return attachments[bloomIndex()]; }
    utils::Attachments& position() { return attachments[positionIndex()]; }
    utils::Attachments& normal() { return attachments[normalIndex()]; }
    utils::Attachments& color() { return attachments[colorIndex()]; }
    utils::Attachments& emission() { return attachments[emissionIndex()]; }
    utils::Attachments& depth() { return attachments[depthIndex()]; }

    std::vector<VkImageView> views(uint32_t frameIndex) const {
        std::vector<VkImageView> views;
        for (const auto& attachment : attachments) {
            views.push_back(attachment.imageView(frameIndex));
        }
        return views;
    }

    std::vector<VkClearValue> clearValues() const {
        std::vector<VkClearValue> values;
        for (const auto& attachment : attachments) {
            values.push_back(attachment.clearValue());
        }
        return values;
    }

    operator std::vector<utils::Attachments>&() {return attachments;}
};

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_RENDER_STAGES_DEFERREDATTACHMENTS_H
