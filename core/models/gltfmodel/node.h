#ifndef GLTFMODEL_NODE_H
#define GLTFMODEL_NODE_H

#include "mesh.h"
#include "skin.h"
#include "quaternion.h"

namespace moon::models {

struct Node {
    Mesh mesh;

    Node* parent{ nullptr };
    Skin* skin{ nullptr };
    std::vector<Node*> children;

    math::Matrix<float, 4, 4> matrix{ 1.0f };
    math::Vector<float, 3> translation{ 0.0f };
    math::Vector<float, 3> scale{ 1.0f };
    math::Quaternion<float> rotation{};

    void update() {
        const auto matrix = getMatrix();
        mesh.uniformBlock.jointcount = skin ? std::min((uint32_t)skin->joints.size(), MAX_NUM_JOINTS) : 0;
        mesh.uniformBlock.matrix = transpose(matrix);

        for (size_t i = 0; i < mesh.uniformBlock.jointcount; i++) {
            mesh.uniformBlock.jointMatrix[i] = transpose(inverse(matrix) * skin->joints[i]->getMatrix() * skin->inverseBindMatrices[i]);
        }
        mesh.uniformBuffer.copy(&mesh.uniformBlock);
    }

    Node(const tinygltf::Node& gltfNode, const tinygltf::Model& gltfModel, const interfaces::Materials& materials, const utils::PhysicalDevice& device, Node* parent, uint32_t& indexStart)
        : parent(parent)
    {
        const auto& nodes = gltfNode;
        convert(translation, nodes.translation);
        convert(rotation, nodes.rotation);
        convert(scale, nodes.scale);
        convert(matrix, nodes.matrix);
        if (const auto meshIndex = nodes.mesh; meshIndex > -1) {
            mesh = Mesh(gltfModel, materials, device, meshIndex, indexStart);
        }
    }

    moon::math::Matrix<float, 4, 4> localMatrix() const {
        return math::translate(translation) * math::rotate(rotation) * math::scale(scale) * matrix;
    }

    moon::math::Matrix<float, 4, 4> getMatrix() const {
        return (parent ? parent->getMatrix() : math::Matrix<float, 4, 4>(1.0f)) * localMatrix();
    }
};

using NodeMap = std::unordered_map<uint32_t, std::unique_ptr<Node>>;

}

#endif