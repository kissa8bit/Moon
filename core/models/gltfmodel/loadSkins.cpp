#include "gltfmodel.h"
#include "node.h"

namespace moon::models {

void GltfModel::loadSkins(const tinygltf::Model& gltfModel) {
    for (auto& instance : instances) {
        instance.skins.reserve(gltfModel.skins.size());
        for (const tinygltf::Skin& source : gltfModel.skins) {
            const auto accessorsIndex = source.inverseBindMatrices;
            if (isInvalid(accessorsIndex)) continue;
            const GltfBufferExtractor extract(gltfModel, accessorsIndex);
            if (!CHECK_M(source.joints.size() == extract.count, "[ GltfModel::loadSkins ] : joints and inverseBindMatrices must be same size")) continue;

            auto& skin = instance.skins.emplace_back();
            for (const auto& jointIndex : source.joints) {
                const auto nodeByIndex = instance.nodes.find(jointIndex);
                if (nodeByIndex == instance.nodes.end()) continue;

                const auto& [_, node] = *nodeByIndex;
                const auto matrix = math::transpose(*((const math::Matrix<float, 4, 4>*)extract.data + skin.size()));
                skin.push_back({ matrix, node.get()});
            }
        }

        for (const auto& [index, node] : instance.nodes) {
            const auto skinIndex = gltfModel.nodes.at(index).skin;
            if (isInvalid(skinIndex)) continue;

            node->skin = &instance.skins.at(skinIndex);
            node->mesh.uniformBlock.jointcount = std::min((uint32_t)node->skin->size(), interfaces::MeshBlock::maxJoints);
        }
    }
}

}