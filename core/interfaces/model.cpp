#include "model.h"
#include "vkdefault.h"
#include "operations.h"

#include "linearAlgebra.h"

namespace moon::interfaces {

void Skeleton::createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout) {
    if (!CHECK_M(VkBuffer(deviceBuffer) != VK_NULL_HANDLE, "[ Skeleton::createDescriptorSet ] deviceBuffer is VK_NULL_HANDLE")) return;

    descriptorSet = descriptorPool.allocateDescriptorSet(descriptorSetLayout);

    VkDescriptorBufferInfo bufferInfo{ deviceBuffer, 0, deviceBuffer.size() };
    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.dstSet = descriptorSet;
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

utils::vkDefault::DescriptorSetLayout Skeleton::descriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

void Mesh::renderBB(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    for (const auto& primitive : primitives) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSets.size(), descriptorSets.data(), 0, NULL);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(math::box), (void*)&primitive.bb);
        vkCmdDraw(commandBuffer, 24, 1, 0, 0);
    }
}

void Mesh::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const {
    for (const auto& primitive : primitives) {
        if (!CHECK_M(primitive.material, std::string("[ Mesh::render ] material is nullptr"))) continue;
        const auto& material = *primitive.material;

        auto descriptors = descriptorSets;
        descriptors.push_back(material.descriptorSet);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

        const auto buffer = material.buffer(primitiveCount++);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(buffer), &buffer);

        if (primitive.index.range.count > 0) {
            vkCmdDrawIndexed(commandBuffer, primitive.index.range.count, 1, primitive.index.range.first, 0, 0);
        }
        else {
            vkCmdDraw(commandBuffer, primitive.vertex.range.count, 1, primitive.vertex.range.first, 0);
        }
    }
}

VkVertexInputBindingDescription Vertex::getBindingDescription(){
    return VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}

std::vector<VkVertexInputAttributeDescription> Vertex::getAttributeDescriptions(){
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, pos)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, normal)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex, uv0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex, uv1)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex, joint0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32A32_SFLOAT,offsetof(Vertex, weight0)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, tangent)});
    attributeDescriptions.push_back(VkVertexInputAttributeDescription{static_cast<uint32_t>(attributeDescriptions.size()),0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, bitangent)});

    return attributeDescriptions;
}

Material::Material(const utils::Texture* emptyTexture) {
    baseColor = emptyTexture;
    metallicRoughness = emptyTexture;
    normal = emptyTexture;
    occlusion = emptyTexture;
    emissive = emptyTexture;
    extensions.specularGlossiness = emptyTexture;
    extensions.diffuse = emptyTexture;
}

void Material::createDescriptorSet(VkDevice device, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout) {
    descriptorSet = descriptorPool.allocateDescriptorSet(descriptorSetLayout);

    auto getDescriptorImageInfo = [](const utils::Texture* tex) {
        VkDescriptorImageInfo descriptorImageInfo{};
        descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptorImageInfo.imageView = tex->imageView();
        descriptorImageInfo.sampler = tex->sampler();
        return descriptorImageInfo;
    };

    VkDescriptorImageInfo baseColorTextureInfo{};
    VkDescriptorImageInfo metallicRoughnessTextureInfo{};

    switch (pbrWorkflows)
    {
        case interfaces::Material::PbrWorkflow::METALLIC_ROUGHNESS: {
            baseColorTextureInfo = getDescriptorImageInfo(baseColor.texture);
            metallicRoughnessTextureInfo = getDescriptorImageInfo(metallicRoughness.texture);
            break;
        }
        case interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS: {
            baseColorTextureInfo = getDescriptorImageInfo(extensions.diffuse.texture);
            metallicRoughnessTextureInfo = getDescriptorImageInfo(extensions.specularGlossiness.texture);
            break;
        }
    }

    std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
        baseColorTextureInfo,
        metallicRoughnessTextureInfo,
        getDescriptorImageInfo(normal.texture),
        getDescriptorImageInfo(occlusion.texture),
        getDescriptorImageInfo(emissive.texture)
    };

    std::vector<VkWriteDescriptorSet> descriptorWrites;
    for (const auto& info : descriptorImageInfos) {
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.back().dstSet = descriptorSet;
        descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size()) - 1;
        descriptorWrites.back().dstArrayElement = 0;
        descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites.back().descriptorCount = 1;
        descriptorWrites.back().pImageInfo = &info;
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

utils::vkDefault::DescriptorSetLayout Material::descriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

Material::Buffer::Buffer(const Material& material, uint32_t primitive) : primitive(primitive) {
    emissiveFactor = material.emissive.factor;
    colorTextureSet = material.baseColor.coordSet;
    normalTextureSet = material.normal.coordSet;
    occlusionTextureSet = material.occlusion.coordSet;
    emissiveTextureSet = material.emissive.coordSet;
    alphaMask = static_cast<float>(material.alphaMode == interfaces::Material::ALPHAMODE_MASK);
    alphaMaskCutoff = material.alphaCutoff;
    workflow = static_cast<float>(material.pbrWorkflows);

    switch (material.pbrWorkflows)
    {
    case interfaces::Material::PbrWorkflow::METALLIC_ROUGHNESS: {
        baseColorFactor = material.baseColor.factor;
        metallicFactor = material.metallicRoughness.factor[Material::metallicIndex];
        roughnessFactor = material.metallicRoughness.factor[Material::roughnessIndex];
        physicalDescriptorTextureSet = material.metallicRoughness.coordSet;
        break;
    }
    case interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS: {
        physicalDescriptorTextureSet = material.extensions.specularGlossiness.coordSet;
        diffuseFactor = material.extensions.diffuse.factor;
        specularFactor = material.extensions.specularGlossiness.factor;
        break;
    }
    }
}

Material::Buffer Material::buffer(uint32_t primitive) const {
    return Buffer(*this, primitive);
}

}
