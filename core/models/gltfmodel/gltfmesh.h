#ifndef GLTFMODEL_MESH_H
#define GLTFMODEL_MESH_H

#include <vector>

#include "matrix.h"
#include "operations.h"
#include "buffer.h"
#include "gltfutils.h"
#include "tinyGLTF.h"

namespace moon::models {

struct gltfMesh : public interfaces::Mesh {
    gltfMesh() = default;
    gltfMesh(const tinygltf::Model& gltfModel, const interfaces::Materials& materials, const utils::PhysicalDevice& device, const size_t meshIndex, uint32_t& firstIndex) {
        uniformBuffer = utils::vkDefault::Buffer(device, device.device(), sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Mesh::Mesh, uniformBuffer");

        for (const tinygltf::Primitive& primitive : gltfModel.meshes[meshIndex].primitives) {
            const auto posAttributes = primitive.attributes.find("POSITION");
            if (posAttributes == primitive.attributes.end()) continue;

            const auto& [_, poseIndex] = *posAttributes;
            const auto& posAccessor = gltfModel.accessors.at(poseIndex);

            const interfaces::Material* material = isValid(primitive.material) ? &materials.at(primitive.material) : nullptr;
            uint32_t indexCount = isValid(primitive.indices) ? gltfModel.accessors.at(primitive.indices).count : 0;
            uint32_t vertexCount = posAccessor.count;

            primitives.emplace_back(
                interfaces::Primitive(firstIndex, indexCount, vertexCount, material, { toVector3f(posAccessor.minValues), toVector3f(posAccessor.maxValues) })
            );
            firstIndex += indexCount;
        }
    };
};

}

#endif