#ifndef MOON_MODELS_GLTFMODEL_INSTANCE_H
#define MOON_MODELS_GLTFMODEL_INSTANCE_H

#include <vector>

#include "node.h"
#include "gltfskeleton.h"
#include "animation.h"

namespace moon::models {

struct Instance {
    Nodes nodes;
    GltfSkeletons skeletons;
    GltfAnimations animations;

    Instance() = default;
    Instance(Instance&& other) noexcept {
        swap(other);
    }
    Instance& operator=(Instance&& other) noexcept {
        swap(other);
        return *this;
    }
    void swap(Instance& other) noexcept {
        std::swap(nodes, other.nodes);
        std::swap(skeletons, other.skeletons);
        std::swap(animations, other.animations);
    }

    Node* loadNode(const tinygltf::Model& gltfModel, uint32_t nodeIndex, Node* parent) {
        Node* current = &(nodes[nodeIndex] = Node(gltfModel.nodes[nodeIndex], parent));
        for (const auto& child : gltfModel.nodes[nodeIndex].children) {
            nodes[nodeIndex].children.push_back(loadNode(gltfModel, child, current));
        }
        return current;
    }
};

using Instances = std::vector<Instance>;

inline void updateNodes(Nodes& nodes, GltfSkeletons& skeletons, bool recursive = true) {
    for (auto& [_, node] : nodes) {
        if (node.parent == nullptr) {
            node.updateMatrix(recursive);
        }
    }
    for (auto& [rootNode, skeleton] : skeletons) {
        skeleton.update(nodes, rootNode);
    }
}

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_INSTANCE_H