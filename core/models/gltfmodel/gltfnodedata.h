#ifndef MOON_MODELS_GLTFMODEL_NODEDATA_H
#define MOON_MODELS_GLTFMODEL_NODEDATA_H

#include <unordered_map>

#include "node.h"
#include "gltfskeleton.h"
#include "gltfmorph.h"

namespace moon::models {

struct GltfNodeData {
    GltfSkeleton skeleton;
    GltfMorphWeight morphWeight;
};
using GltfNodeDataMap = std::unordered_map<Node::Id, GltfNodeData>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_NODEDATA_H
