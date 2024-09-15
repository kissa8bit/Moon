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

void Object::setEnable(const bool& inenable) {
    enable = inenable;
}

void Object::setEnableShadow(const bool& inenable) {
    enableShadow = inenable;
}

bool Object::outlining() const {
    return (pipelineBitMask & interfaces::ObjectProperty::outlining);
}

Object::Object(uint8_t pipelineBitMask)
    : pipelineBitMask(pipelineBitMask)
{}

Object::Object(uint8_t pipelineBitMask, Model* model, uint32_t firstInstance, uint32_t instanceCount)
    : pipelineBitMask(pipelineBitMask), pModel(model), firstInstance(firstInstance), instanceCount(instanceCount)
{}

Model* Object::model() {
    return pModel;
}

uint32_t Object::getInstanceNumber(uint32_t imageNumber) const {
    return firstInstance + (instanceCount > imageNumber ? imageNumber : 0);
}

uint8_t& Object::pipelineFlagBits() {
    return pipelineBitMask;
}

void Object::setFirstPrimitive(uint32_t infirstPrimitive) {
    firstPrimitive = infirstPrimitive;
}

void Object::setPrimitiveCount(uint32_t inprimitiveCount) {
    primitiveCount = inprimitiveCount;
}

bool Object::comparePrimitive(uint32_t primitive) {
    return !(primitive < firstPrimitive) && (primitive < firstPrimitive + primitiveCount);
}

uint32_t Object::getFirstPrimitive() const {
    return firstPrimitive;
}

uint32_t Object::getPrimitiveCount() const {
    return primitiveCount;
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
