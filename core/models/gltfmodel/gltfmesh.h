#ifndef GLTFMODEL_MESH_H
#define GLTFMODEL_MESH_H

#include "linearAlgebra.h"
#include "tinyGLTF.h"
#include "node.h"
#include "skin.h"

namespace moon::models {

struct GltfMesh : public interfaces::Mesh {
    GltfMesh() = default;
    GltfMesh(const tinygltf::Model& gltfModel, const tinygltf::Mesh& mesh, const interfaces::Materials& materials, uint32_t& firstIndex) {
        for (const tinygltf::Primitive& primitive : mesh.primitives) {
            const auto posAttributes = primitive.attributes.find("POSITION");
            if (posAttributes == primitive.attributes.end()) continue;

            const auto& [_, poseIndex] = *posAttributes;
            const auto& posAccessor = gltfModel.accessors.at(poseIndex);

            const interfaces::Material* material = isValid(primitive.material) ? &materials.at(primitive.material) : &materials.back();
            const uint32_t indexCount = isValid(primitive.indices) ? gltfModel.accessors.at(primitive.indices).count : 0;
            const uint32_t vertexCount = posAccessor.count;

            primitives.emplace_back(
                interfaces::Primitive({firstIndex, indexCount}, {0, vertexCount}, material, { toVector3f(posAccessor.minValues), toVector3f(posAccessor.maxValues) })
            );
            firstIndex += indexCount;
        }
    };
};

using GltfMeshes = std::unordered_map<Node::Id, GltfMesh>;

}

#endif