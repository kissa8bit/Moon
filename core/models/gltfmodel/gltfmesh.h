#ifndef GLTFMODEL_MESH_H
#define GLTFMODEL_MESH_H

#include <vector>

#include "linearAlgebra.h"

#include "operations.h"
#include "buffer.h"
#include "gltfutils.h"
#include "tinyGLTF.h"
#include "skin.h"

namespace moon::models {

struct GltfMesh : public interfaces::Mesh {
    Skin* skin{ nullptr };

    GltfMesh() = default;
    GltfMesh(const tinygltf::Model& gltfModel, const interfaces::Materials& materials, const size_t meshIndex, uint32_t& firstIndex) {
        for (const tinygltf::Primitive& primitive : gltfModel.meshes[meshIndex].primitives) {
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

using GltfMeshes = std::unordered_map<NodeId, GltfMesh>;

}

#endif