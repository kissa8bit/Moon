#include "model.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::interfaces {

BoundingBox::BoundingBox(moon::math::Vector<float,3> min, moon::math::Vector<float,3> max) : min(min), max(max) {};

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

moon::utils::vkDefault::DescriptorSetLayout Model::createNodeDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

moon::utils::vkDefault::DescriptorSetLayout Model::createMaterialDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

}
