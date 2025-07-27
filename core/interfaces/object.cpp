#include "object.h"
#include "vkdefault.h"
#include "operations.h"

namespace moon::interfaces {

bool Object::getEnable() const {
    return enable;
}

bool Object::getEnableShadow() const {
    return enableShadow;
}

void Object::setEnable(const bool inenable) {
    enable = inenable;
}

void Object::setEnableShadow(const bool inenable) {
    enableShadow = inenable;
}

Object::Object(ObjectMask objectMask, Model* model, const Range& instanceRange)
    : mask(objectMask), pModel(model), instance({ instanceRange })
{}

Model* Object::model() {
    return pModel;
}

uint32_t Object::getInstanceNumber(uint32_t imageNumber) const {
    return instance.range.first + (instance.range.first > imageNumber ? imageNumber : 0);
}

ObjectMask& Object::objectMask() {
    return mask;
}

Range& Object::primitiveRange() { return primitive.range;}

bool Object::comparePrimitive(uint32_t primitiveIndex) const {
    return !(primitiveIndex < primitive.range.first) && (primitiveIndex < primitive.range.last());
}

const VkDescriptorSet& Object::getDescriptorSet(uint32_t i) const {
    return descriptors[i];
}

utils::vkDefault::DescriptorSetLayout Object::createBaseDescriptorSetLayout(VkDevice device){
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

utils::vkDefault::DescriptorSetLayout Object::createSkyboxDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
        binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
        binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

}
