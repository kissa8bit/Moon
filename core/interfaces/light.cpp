#include "light.h"

#include <utils/operations.h>

namespace moon::interfaces {

Light::Light(LightMask lightMask)
    : mask(lightMask)
{}

const VkDescriptorSet& Light::getDescriptorSet(utils::ResourceIndex resourceIndex) const {
    return descriptorSets[resourceIndex.get()];
}

LightMask& Light::lightMask() {
    return mask;
}

const LightMask& Light::lightMask() const {
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
