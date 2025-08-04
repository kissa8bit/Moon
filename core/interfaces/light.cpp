#include "light.h"

#include <utils/operations.h>

namespace moon::interfaces {

Light::Light(LightMask lightMask)
    : mask(lightMask)
{}

const VkDescriptorSet& Light::getDescriptorSet(uint32_t i) const {
    return descriptorSets[i];
}

LightMask& Light::lightMask() {
    return mask;
}

utils::vkDefault::DescriptorSetLayout Light::createDescriptorSetLayout(VkDevice device){
    utils::vkDefault::DescriptorSetLayout descriptorSetLayout;

    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, binding);
    return descriptorSetLayout;
}

}
