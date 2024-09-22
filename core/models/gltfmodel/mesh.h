#ifndef GLTFMODEL_MESH_H
#define GLTFMODEL_MESH_H

#include <vector>

#include "primitive.h"
#include "matrix.h"
#include "operations.h"
#include "buffer.h"
#include "gltfutils.h"
#include "tinyGLTF.h"

namespace moon::models {

struct Mesh {
    utils::Buffer uniformBuffer;
    interfaces::MeshBlock uniformBlock;

    std::vector<Primitive> primitives;
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

    Mesh() = default;
    Mesh(const tinygltf::Model& gltfModel, const interfaces::Materials& materials, const utils::PhysicalDevice& device, const size_t meshIndex, uint32_t& firstIndex) {
        uniformBuffer = utils::vkDefault::Buffer(device, device.device(), sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", Mesh::Mesh, uniformBuffer");

        for (const tinygltf::Primitive& primitive : gltfModel.meshes[meshIndex].primitives) {
            const auto posAttributes = primitive.attributes.find("POSITION");
            if (posAttributes == primitive.attributes.end()) {
                continue;
            }
            const auto& [_, poseIndex] = *posAttributes;
            const auto& posAccessor = gltfModel.accessors.at(poseIndex);

            const interfaces::Material* material = isValid(primitive.material) ? &materials.at(primitive.material) : nullptr;
            uint32_t indexCount = isValid(primitive.indices) ? gltfModel.accessors.at(primitive.indices).count : 0;
            uint32_t vertexCount = posAccessor.count;

            primitives.emplace_back(
                Primitive(firstIndex, indexCount, vertexCount, material, { toVector3f(posAccessor.minValues), toVector3f(posAccessor.maxValues) })
            );
            firstIndex += indexCount;
        }
    };

    bool empty() const {
        return VkBuffer(uniformBuffer) == VK_NULL_HANDLE;
    }

    void createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout) {
        if (empty()) return;

        descriptorSet = descriptorPool.allocateDescriptorSet(descriptorSetLayout);

        VkDescriptorBufferInfo bufferInfo{ uniformBuffer, 0, sizeof(interfaces::MeshBlock) };
        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
    }
};

}

#endif