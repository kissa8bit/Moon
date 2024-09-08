#include "model.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::interfaces {

BoundingBox::BoundingBox(math::Vector<float,3> min, math::Vector<float,3> max) : min(min), max(max) {};

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

utils::vkDefault::DescriptorSetLayout Model::createNodeDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

utils::vkDefault::DescriptorSetLayout Model::createMaterialDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

Material::Material(const utils::Texture* empty) {
    baseColorTexture = empty;
    metallicRoughnessTexture = empty;
    normalTexture = empty;
    occlusionTexture = empty;
    emissiveTexture = empty;
    extension.specularGlossinessTexture = empty;
    extension.diffuseTexture = empty;
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
    if (pbrWorkflows.metallicRoughness) {
        baseColorTextureInfo = getDescriptorImageInfo(baseColorTexture);
    }
    if (pbrWorkflows.specularGlossiness) {
        baseColorTextureInfo = getDescriptorImageInfo(extension.diffuseTexture);
    }

    VkDescriptorImageInfo metallicRoughnessTextureInfo{};
    if (pbrWorkflows.metallicRoughness) {
        metallicRoughnessTextureInfo = getDescriptorImageInfo(metallicRoughnessTexture);
    }
    if (pbrWorkflows.specularGlossiness) {
        metallicRoughnessTextureInfo = getDescriptorImageInfo(extension.specularGlossinessTexture);
    }

    std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
        baseColorTextureInfo,
        metallicRoughnessTextureInfo,
        getDescriptorImageInfo(normalTexture),
        getDescriptorImageInfo(occlusionTexture),
        getDescriptorImageInfo(emissiveTexture)
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

}
