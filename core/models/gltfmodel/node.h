#ifndef GLTFMODEL_NODE_H
#define GLTFMODEL_NODE_H

#include "linearAlgebra.h"

#include <numeric>

namespace moon::models {

struct Node {
    Node* parent{ nullptr };
    ChildrenNodes children;

    math::mat4 matrix{ 1.0f };
    math::mat4 global{ 1.0f };
    math::vec3 translation{ 0.0f };
    math::vec3 scale{ 1.0f };
    math::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };

    Node(const tinygltf::Node& gltfNode, Node* parent)
        : parent(parent)
    {
        convert(translation, gltfNode.translation);
        convert(rotation, gltfNode.rotation);
        convert(scale, gltfNode.scale);
        convert(global, gltfNode.matrix);
    }

    moon::math::mat4 localMatrix() const {
        return math::translate(translation) * math::rotate(rotation) * math::scale(scale) * global;
    }

    void updateMatrix(bool recursive = false) {
        matrix = (parent ? parent->matrix : math::mat4::identity()) * localMatrix();
        if (recursive) for (auto child : children) child->updateMatrix(recursive);
    }

};

inline void updateRootNodes(const RootNodes& rootNodes, bool recursive = true) {
    for (const auto& node : rootNodes) {
        node->updateMatrix(recursive);
    }
}

}

#endif