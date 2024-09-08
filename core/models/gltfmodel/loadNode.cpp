#include "gltfmodel.h"

namespace moon::models {

void GltfModel::loadNode(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, NodeMap& nodeMap, Node* parent, uint32_t nodeIndex, uint32_t& indexStart) {
    nodeMap[nodeIndex] = std::make_unique<Node>(gltfModel.nodes[nodeIndex], gltfModel, materials, device, parent, indexStart);

    for (const auto& children : gltfModel.nodes[nodeIndex].children) {
        loadNode(gltfModel, device, nodeMap, nodeMap[nodeIndex].get(), children, indexStart);
    }
}

}