#include "light.h"
#include "vkdefault.h"
#include "operations.h"
#include <vector>

namespace moon::interfaces {

Light::Light(uint8_t pipelineBitMask, bool enableShadow, bool enableScattering)
    : pipelineBitMask(pipelineBitMask), enableShadow(enableShadow), enableScattering(enableScattering) {}

void Light::setEnableShadow(bool enable){
    enableShadow = enable;
}

void Light::setEnableScattering(bool enable){
    enableScattering = enable;
}

bool Light::isShadowEnable() const{
    return enableShadow;
}

bool Light::isScatteringEnable() const{
    return enableScattering;
}

const VkDescriptorSet& Light::getDescriptorSet(uint32_t i) const {
    return descriptorSets[i];
}

uint8_t& Light::pipelineFlagBits() {
    return pipelineBitMask;
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
