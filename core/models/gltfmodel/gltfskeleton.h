#ifndef GLTFSKELETON_H
#define GLTFSKELETON_H

#include <unordered_map>

#include "skin.h"
#include "model.h"
#include "gltfutils.h"

namespace moon::models {

struct GltfSkeleton : public interfaces::Skeleton {
    const Skin* skin{ nullptr };

    void update(const NodeMap& nodeMap, NodeId rootNode) {
        hostBuffer.matrix = transpose(nodeMap.at(rootNode)->matrix);
        if (skin) {
            const auto& joints = (*skin);
            for (size_t i = 0; i < joints.size(); i++) {
                hostBuffer.jointMatrix[i] = transpose(inverse(hostBuffer.matrix) * nodeMap.at(joints[i].jointedNodeId)->matrix * joints[i].inverseBindMatrices);
            }
        }
        deviceBuffer.copy(&hostBuffer);
    }
};

using GltfSkeletons = std::unordered_map<NodeId, GltfSkeleton>;

}

#endif