#include "gltfmodel.h"

namespace moon::models {

Node* GltfModel::loadNode(const tinygltf::Model& gltfModel, NodeMap& nodeMap, Node* parent, uint32_t nodeIndex, uint32_t& indexStart) {
    nodeMap[nodeIndex] = std::make_unique<Node>(gltfModel.nodes[nodeIndex], gltfModel, materials, parent, indexStart);

    for (const auto& children : gltfModel.nodes[nodeIndex].children) {
        nodeMap[nodeIndex]->children.push_back(loadNode(gltfModel, nodeMap, nodeMap[nodeIndex].get(), children, indexStart));
    }
    return nodeMap[nodeIndex].get();
}

}