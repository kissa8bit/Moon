#ifndef MOON_MODELS_GLTFMODEL_MESH_H
#define MOON_MODELS_GLTFMODEL_MESH_H

#include <math/linearAlgebra.h>

#include "tinyGLTF.h"
#include "node.h"
#include "skin.h"

namespace moon::models {

struct GltfMesh : public interfaces::Mesh {
    Boxes boxes;

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
                interfaces::Primitive({firstIndex, indexCount}, {0, vertexCount}, material, { math::vec4(toVector3f(posAccessor.minValues), -1.0f), math::vec4(toVector3f(posAccessor.maxValues), -1.0f) })
            );
            firstIndex += indexCount;
        }
    };

    void calculateNodeBoxes(const interfaces::Vertices& vertices, const interfaces::Indices& indices, const Skin& skin) {
        for (const auto& primitive : primitives) {
            const auto& range = primitive.index.range;
            for (uint32_t index = range.first; index < range.last(); index++) {
                const auto& vertex = vertices[indices[index]];
                const auto& pos = vertex.pos;
                const int joint = vertex.joint[0];
                if (joint == -1) continue;

                auto& box = boxes[skin.joints.at(joint).jointedNodeId];
                box.max = math::vec4(math::max(box.max.dvec(), pos), joint);
                box.min = math::vec4(math::min(box.min.dvec(), pos), joint);
            }
        }
    }

    void renderNodeBB(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
        for (const auto& [_, box] : boxes) {
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSets.size(), descriptorSets.data(), 0, NULL);
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(math::box), (void*)&box);
            vkCmdDraw(commandBuffer, 24, 1, 0, 0);
        }
    }
};

using GltfMeshes = std::unordered_map<Node::Id, GltfMesh>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_MESH_H