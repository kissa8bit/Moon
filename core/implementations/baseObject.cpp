#include "baseObject.h"

namespace moon::implementations {

utils::vkDefault::DescriptorSetLayout BaseObject::createDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.back().stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

BaseObject::BaseObject(interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount)
    :   interfaces::Object(
            interfaces::ObjectMask(
                interfaces::objectTypeFromVertexType(model ? model->vertexType() : interfaces::Model::VertexType{}),
                interfaces::ObjectProperty::enable),
            model,
            { firstInstance, instanceCount }
        ),
        uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void BaseObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseObject::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = createDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
    for (size_t i = 0; i < imageCount; i++) {
        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptors.at(i), uniformBuffer.device.at(i).descriptorBufferInfo());
        utils::descriptorSet::update(device.device(), writes);
    }
}

void BaseObject::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
    createDescriptors(device, imageCount);
}

utils::Buffers& BaseObject::buffers() {
    return uniformBuffer.device;
}

BaseObject::Buffer& BaseObject::buffer(bool markDirty) {
    if (markDirty) {
        utils::vkDefault::raiseFlags(uniformBuffer.device);
    }
    return hostBuffer;
}

void BaseObject::setTransformation(const math::mat4& transformation) {
    buffer(true).modelMatrix = transpose(transformation);
}

} // moon::implementations