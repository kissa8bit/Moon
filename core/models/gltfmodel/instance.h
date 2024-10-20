#ifndef GLTFMODEL_INSTANCE_H
#define GLTFMODEL_INSTANCE_H

#include <vector>

#include "node.h"
#include "gltfskeleton.h"
#include "animation.h"

namespace moon::models {

struct Instance {
    NodeMap nodes;
    RootNodes rootNodes;
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
        std::swap(rootNodes, other.rootNodes);
        std::swap(skeletons, other.skeletons);
        std::swap(animations, other.animations);
    }

    Node* loadNode(const tinygltf::Model& gltfModel, uint32_t nodeIndex, Node* parent) {
        nodes[nodeIndex] = std::make_unique<Node>(gltfModel.nodes[nodeIndex], parent);
        Node* node = nodes[nodeIndex].get();

        for (const auto& child : gltfModel.nodes[nodeIndex].children) {
            nodes[nodeIndex]->children.push_back(loadNode(gltfModel, child, node));
        }
        return node;
    }
};

using Instances = std::vector<Instance>;

}

#endif