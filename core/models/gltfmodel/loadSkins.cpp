#include "gltfmodel.h"
#include "node.h"
#include "skin.h"

namespace moon::models {

void GltfModel::loadSkins(const tinygltf::Model& gltfModel) {
    skins.reserve(gltfModel.skins.size());
    for (const tinygltf::Skin& source : gltfModel.skins) {
        const auto accessorsIndex = source.inverseBindMatrices;
        if (isInvalid(accessorsIndex)) continue;
        const GltfBufferExtractor extract(gltfModel, accessorsIndex);
        if (!CHECK_M(source.joints.size() == extract.count, "[ GltfModel::loadSkins ] : joints and inverseBindMatrices must be same size")) continue;

        auto& skin = skins.emplace_back();
        for (const auto& jointIndex : source.joints) {
            const auto matrix = math::transpose(*((const math::mat4*)extract.data + skin.size()));
            skin.push_back({ matrix, static_cast<uint32_t>(jointIndex) });
        }
    }

    for (Node::Id nodeId = 0; nodeId < gltfModel.nodes.size(); nodeId++) {
        if (const auto skinIndex = gltfModel.nodes[nodeId].skin; !isInvalid(skinIndex)) {
            meshes[nodeId].skin = &skins.at(skinIndex);
        }
    }
}

}