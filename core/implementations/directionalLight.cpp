#include "directionalLight.h"

#include <utils/operations.h>

namespace moon::implementations {

DirectionalLight::DirectionalLight()
    : Light(interfaces::LightMask(interfaces::LightType::directional)), uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void DirectionalLight::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

    VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
    texture = utils::Texture::createEmpty(device, commandBuffer);
    CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
    texture.destroyCache();

    createDescriptors(device, imageCount);
}

void DirectionalLight::setEnable(bool enable) { enabled = enable; }
bool DirectionalLight::isEnable() const { return enabled; }

void DirectionalLight::render(
    utils::ResourceIndex resourceIndex,
    VkCommandBuffer commandBuffer,
    const utils::vkDefault::DescriptorSets& descriptorSet,
    VkPipelineLayout pipelineLayout,
    VkPipeline pipeline)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    utils::vkDefault::DescriptorSets descriptors = descriptorSet;
    descriptors.push_back(descriptorSets[resourceIndex.get()]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptors.size()), descriptors.data(), 0, nullptr);
    if (enabled) {
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
}

void DirectionalLight::update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(resourceIndex, commandBuffer);
}

void DirectionalLight::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Light::createDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSets.at(i), uniformBuffer.device.at(i).descriptorBufferInfo());
        WRITE_DESCRIPTOR(writes, descriptorSets.at(i), texture.descriptorImageInfo());
        utils::descriptorSet::update(device.device(), writes);
    }
}

utils::Buffers& DirectionalLight::buffers() {
    return uniformBuffer.device;
}

DirectionalLight::Buffer& DirectionalLight::buffer(bool markDirty) {
    if (markDirty) {
        utils::vkDefault::raiseFlags(uniformBuffer.device);
    }
    return hostBuffer;
}

const DirectionalLight::Buffer& DirectionalLight::buffer() const {
    return hostBuffer;
}

void DirectionalLight::setTransformation(const math::mat4& transformation) {
    buffer(true).view = transpose(inverse(transformation));
}

} // moon::implementations
