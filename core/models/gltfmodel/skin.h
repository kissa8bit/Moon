#ifndef MOON_MODELS_GLTFMODEL_SKIN_H
#define MOON_MODELS_GLTFMODEL_SKIN_H

#include <vector>
#include <unordered_map>

#include <math/linearAlgebra.h>

#include "node.h"

namespace moon::models {

struct Skin {
    struct Joint {
        math::mat4 inverseBindMatrices{ math::mat4::identity() };
        Node::Id jointedNodeId{ Node::invalidId };
    };
    using Joints = std::vector<Joint>;

    Joints joints;

    Skin() = default;
    Skin(const tinygltf::Model& gltfModel, const tinygltf::Skin& skin) {
        const auto accessorsIndex = skin.inverseBindMatrices;
        if (isInvalid(accessorsIndex)) return;
        const GltfBufferExtractor extract(gltfModel, accessorsIndex);
        if (!CHECK_M(skin.joints.size() == extract.count, "[ Skin::Skin ] : joints and inverseBindMatrices must be same size")) return;

        for (const auto& jointIndex : skin.joints) {
            const auto matrix = math::transpose(*((const math::mat4*)extract.data + joints.size()));
            joints.push_back({ matrix, static_cast<uint32_t>(jointIndex) });
        }
    }
};

using Skins = std::unordered_map<Node::Id, Skin>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_SKIN_H