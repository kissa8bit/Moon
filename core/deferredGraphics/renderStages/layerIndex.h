#ifndef MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERINDEX_H
#define MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERINDEX_H

#include <utils/types.h>

namespace moon::deferredGraphics {

struct LayerIndexTag {};
using LayerIndex = utils::StrongIndex<LayerIndexTag>;

template <typename T = utils::AttachmentName>
T layerPrefix(LayerIndex layerIndex) {
    return T("l" + std::to_string(layerIndex.get()) + ".");
}

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_RENDER_STAGES_LAYERINDEX_H