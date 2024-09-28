#ifndef GLTFMODEL_NODE_H
#define GLTFMODEL_NODE_H

#include "gltfmesh.h"
#include "skin.h"
#include "quaternion.h"

namespace moon::models {

struct Node {
    GltfMesh mesh;

    Node* parent{ nullptr };
    Skin* skin{ nullptr };
    std::vector<Node*> children;

    math::Matrix<float, 4, 4> matrix{ 1.0f };
    math::Matrix<float, 4, 4> global{ 1.0f };
    math::Vector<float, 3> translation{ 0.0f };
    math::Vector<float, 3> scale{ 1.0f };
    math::Quaternion<float> rotation{ 1.0f, 0.0f, 0.0f, 0.0f };

    Node(const tinygltf::Node& gltfNode, const tinygltf::Model& gltfModel, const interfaces::Materials& materials, Node* parent, uint32_t& indexStart)
        : parent(parent)
    {
        convert(translation, gltfNode.translation);
        convert(rotation, gltfNode.rotation);
        convert(scale, gltfNode.scale);
        convert(global, gltfNode.matrix);
        if (const auto meshIndex = gltfNode.mesh; isValid(meshIndex)) {
            mesh = GltfMesh(gltfModel, materials, meshIndex, indexStart);
        }
    }

    moon::math::Matrix<float, 4, 4> localMatrix() const {
        return math::translate(translation) * math::rotate(rotation) * math::scale(scale) * global;
    }

    void updateMatrix(bool recursive = false) {
        matrix = (parent ? parent->matrix : math::Matrix<float, 4, 4>(1.0f)) * localMatrix();
        if (recursive) for (auto child : children) child->updateMatrix(recursive);
    }

    void updateMesh(bool recursive = false) {
        mesh.uniformBlock.matrix = transpose(matrix);
        for (size_t i = 0; i < mesh.uniformBlock.jointcount; i++) {
            mesh.uniformBlock.jointMatrix[i] = transpose(inverse(matrix) * skin->joints[i]->matrix * skin->inverseBindMatrices[i]);
        }
        mesh.uniformBuffer.copy(&mesh.uniformBlock);
        if(recursive) for (auto child : children) child->updateMesh(recursive);
    }
};

using NodeMap = std::unordered_map<uint32_t, std::unique_ptr<Node>>;
using RootNodes = std::vector<Node*>;

inline void updateRootNodes(const RootNodes& rootNodes, bool recursive = true) {
    for (const auto& node : rootNodes) {
        node->updateMatrix(recursive);
    }
    for (const auto& node : rootNodes) {
        node->updateMesh(recursive);
    }
}

}

#endif