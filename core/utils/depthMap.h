#ifndef MOON_UTILS_DEPTHMAP_H
#define MOON_UTILS_DEPTHMAP_H

#include "attachments.h"
#include "device.h"
#include "texture.h"

namespace moon::utils {

class DepthMap {
private:
    struct Map {
        utils::Attachments attachments;
        utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
        utils::vkDefault::DescriptorPool descriptorPool;
        utils::vkDefault::DescriptorSets descriptorSets;

        Map() = default;
        Map(const PhysicalDevice& device, const utils::vkDefault::ImageInfo& imageInfo);
    } map;

    Texture emptyTextureWhite;
    utils::vkDefault::ImageInfo imageInfo;
    VkDevice device{VK_NULL_HANDLE};

public:
    DepthMap() = default;
    DepthMap(const PhysicalDevice& device, VkCommandPool commandPool, const utils::vkDefault::ImageInfo& imageInfo);
    void update(bool enable);

    const utils::vkDefault::DescriptorSets& descriptorSets() const;
    const utils::Attachments& attachments() const;

    static moon::utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);
};

}
#endif // MOON_UTILS_DEPTHMAP_H
