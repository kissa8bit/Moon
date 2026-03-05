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
        utils::vkDefault::depthSampler())),
    descriptorSetLayout(DepthMap::createDescriptorSetLayout(device.device())),
    descriptorPool(utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageInfo.Count)),
    descriptorSets(descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageInfo.Count))
{}

DepthMap::DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, const utils::vkDefault::ImageInfo& imageInfo) : 
    map(device, imageInfo), 
    imageInfo(imageInfo), 
    device(device.device())
{}

const utils::vkDefault::DescriptorSets& DepthMap::descriptorSets() const{
    return map.descriptorSets;
}

void DepthMap::update(){
    CHECK_M(map.descriptorSets.size() == imageInfo.Count, std::string("[ DepthMap::update ] descriptorSets.size() mismatch with imageInfo.Count"));
    CHECK_M(map.attachments.count() == imageInfo.Count, std::string("[ DepthMap::update ] attachments.count() mismatch with imageInfo.Count"));

    for (uint32_t i = 0; i < imageInfo.Count; i++) {
        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, map.descriptorSets[i], map.attachments.descriptorImageInfo(i));
        utils::descriptorSet::update(device, writes);
    }
}

const Attachments& DepthMap::attachments() const {
    return map.attachments;
}

moon::utils::vkDefault::DescriptorSetLayout DepthMap::createDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

}
