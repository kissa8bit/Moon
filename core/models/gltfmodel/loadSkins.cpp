#include "gltfmodel.h"
#include "node.h"

namespace moon::models {

void GltfModel::loadSkins(const tinygltf::Model& gltfModel) {
    for (auto& instance : instances) {
        instance.skins.reserve(gltfModel.skins.size());
        for (const tinygltf::Skin& source : gltfModel.skins) {
            auto& skin = instance.skins.emplace_back();

            for (const auto& jointIndex : source.joints) {
                const auto nodeByIndex = instance.nodes.find(jointIndex);
                if (nodeByIndex == instance.nodes.end()) continue;

                const auto& [_, node] = *nodeByIndex;
                skin.joints.push_back(node.get());
            }

            const auto accessorsIndex = source.inverseBindMatrices;
            if (isInvalid(accessorsIndex)) continue;
            const auto [data, count, _] = extractBuffer(gltfModel, accessorsIndex);

            auto& matrices = skin.inverseBindMatrices;
            matrices.resize(count);
            std::memcpy((void*)matrices.data(), data, count * sizeof(math::Matrix<float, 4, 4>));
            std::transform(matrices.begin(), matrices.end(), matrices.begin(), [](const auto& matrix) {
                return math::transpose(matrix);
            });
        }

        for (const auto& [index, node] : instance.nodes) {
            const auto skinIndex = gltfModel.nodes.at(index).skin;
            if (isInvalid(skinIndex)) continue;

            node->skin = &instance.skins.at(skinIndex);
        }
    }
}

}