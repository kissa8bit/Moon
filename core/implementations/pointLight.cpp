#include "pointLight.h"

namespace moon::implementations {

static constexpr uint32_t kSphereVertexCount = 8 * 8 * 6;

PointLight::PointLight()
    : Light(interfaces::LightMask(interfaces::LightType::pointLight)), uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void PointLight::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

    VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
    texture = utils::Texture::createEmpty(device, commandBuffer);
    CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
    texture.destroyCache();

    createDescriptors(device, imageCount);
}

void PointLight::render(
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
    vkCmdDraw(commandBuffer, kSphereVertexCount, 1, 0, 0);
}

void PointLight::update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(resourceIndex, commandBuffer);
}

void PointLight::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
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

utils::Buffers& PointLight::buffers() {
    return uniformBuffer.device;
}

PointLight::Buffer& PointLight::buffer(bool markDirty) {
    if (markDirty) {
        utils::vkDefault::raiseFlags(uniformBuffer.device);
    }
    return hostBuffer;
}

const PointLight::Buffer& PointLight::buffer() const {
    return hostBuffer;
}

void PointLight::setTransformation(const math::mat4& transformation) {
    buffer(true).position = transformation * math::vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

} // moon::implementations
