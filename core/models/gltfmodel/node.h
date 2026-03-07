#ifndef MOON_MODELS_GLTFMODEL_NODE_H
#define MOON_MODELS_GLTFMODEL_NODE_H

#include <numeric>

#include <math/linearAlgebra.h>

#include "gltfutils.h"

namespace moon::models {

struct Node {
    using Id = uint32_t;
    using Children = std::vector<Node*>;
    const static Id invalidId = std::numeric_limits<Id>::max();

    Node* parent{ nullptr };
    Children children;

    math::mat4 matrix{ 1.0f };
    math::mat4 global{ 1.0f };
    math::vec3 translation{ 0.0f };
    math::vec3 scale{ 1.0f };
    math::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
    std::vector<float> weights;

    math::vec3 restTranslation{ 0.0f };
    math::vec3 restScale{ 1.0f };
    math::quat restRotation{ 1.0f, 0.0f, 0.0f, 0.0f };
    std::vector<float> restWeights;

    Node() = default;
    Node(const tinygltf::Node& gltfNode, Node* parent)
        : parent(parent)
    {
        convert(translation, gltfNode.translation);
        convert(rotation, gltfNode.rotation);
        convert(scale, gltfNode.scale);
        convert(global, gltfNode.matrix);
        restTranslation = translation;
        restScale = scale;
        restRotation = rotation;
        for (double w : gltfNode.weights) restWeights.push_back(static_cast<float>(w));
        weights = restWeights;
    }

    moon::math::mat4 localMatrix() const {
        return math::translate(translation) * math::rotate(rotation) * math::scale(scale) * global;
    }

    void updateMatrix(bool recursive = false) {
        matrix = (parent ? parent->matrix : math::mat4::identity()) * localMatrix();
        if (recursive) for (auto child : children) child->updateMatrix(recursive);
    }
};

using Nodes = std::unordered_map<Node::Id, Node>;
using Boxes = std::unordered_map<Node::Id, math::box>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_NODE_H