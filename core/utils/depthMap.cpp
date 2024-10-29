#include "depthMap.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::utils {

DepthMap::Map::Map(const PhysicalDevice& device, const utils::vkDefault::ImageInfo& imageInfo) :
    attachments(utils::Attachments(
        device,
        device.device(),
        imageInfo,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        { {1.0,0} },
        utils::vkDefault::sampler())),
    descriptorSetLayout(DepthMap::createDescriptorSetLayout(device.device())),
    descriptorPool(utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageInfo.Count)),
    descriptorSets(descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count))
{}

DepthMap::DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, const utils::vkDefault::ImageInfo& imageInfo)
    : map(device, imageInfo), emptyTextureWhite(utils::Texture::empty(device, commandPool, false)), imageInfo(imageInfo), device(device.device())
{}

const utils::vkDefault::DescriptorSets& DepthMap::descriptorSets() const{
    return map.descriptorSets;
}

void DepthMap::update(bool enable){
    for (size_t i = 0; i < imageInfo.Count; i++)
    {
        VkDescriptorImageInfo shadowImageInfo{};
            shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            shadowImageInfo.imageView = enable && map.attachments.count() ? map.attachments.imageView(i) : emptyTextureWhite.imageView();
            shadowImageInfo.sampler = enable && map.attachments.count() ? map.attachments.sampler() : emptyTextureWhite.sampler();
        std::vector<VkWriteDescriptorSet> descriptorWrites;
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = map.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &shadowImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

const Attachments& DepthMap::attachments() const {
    return map.attachments;
}

moon::utils::vkDefault::DescriptorSetLayout DepthMap::createDescriptorSetLayout(VkDevice device) {
    moon::utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, binding);
    return descriptorSetLayout;
}

}
