#include "camera.h"

#include <utils/operations.h>

namespace moon::interfaces {

const VkDescriptorSet& Camera::getDescriptorSet(utils::ResourceIndex resourceIndex) const {
    return descriptorSets[resourceIndex.get()];
}

utils::vkDefault::DescriptorSetLayout Camera::createDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

} // moon::interfaces
