#include "gltfmodel.h"
#include "gltfmodel/node.h"

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
            if (accessorsIndex == -1) continue;

            const tinygltf::Accessor& accessor = gltfModel.accessors[accessorsIndex];
            const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = gltfModel.buffers[bufferView.buffer];
            const void* data = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

            auto& matrices = skin.inverseBindMatrices;
            matrices.resize(accessor.count);
            std::memcpy((void*)matrices.data(), data, accessor.count * sizeof(math::Matrix<float, 4, 4>));
            std::transform(matrices.begin(), matrices.end(), matrices.begin(), [](const auto& matrix) {
                return math::transpose(matrix);
                });
        }

        for (const auto& [index, node] : instance.nodes) {
            const auto skinIndex = gltfModel.nodes.at(index).skin;
            if (skinIndex == -1) continue;

            node->skin = &instance.skins.at(skinIndex);
        }
    }
}

}