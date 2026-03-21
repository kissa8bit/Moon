#include "baseCamera.h"

#include <utils/operations.h>

namespace moon::implementations {

BaseCamera::BaseCamera()
    : uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void BaseCamera::update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) {
    hostBuffer.invViewProj = math::inverse(hostBuffer.view * hostBuffer.proj);
    uniformBuffer.update(resourceIndex, commandBuffer);
}

void BaseCamera::create(const utils::PhysicalDevice& device, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
    createDescriptors(device, imageCount);
}

void BaseCamera::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Camera::createDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSets.at(i), uniformBuffer.device.at(i).descriptorBufferInfo());
        utils::descriptorSet::update(device.device(), writes);
    }
}

utils::Buffers& BaseCamera::buffers() {
    return uniformBuffer.device;
}

BaseCamera::Buffer& BaseCamera::buffer(bool markDirty) {
	if (markDirty) {
		utils::vkDefault::raiseFlags(uniformBuffer.device);
	}
	return hostBuffer;
}

const BaseCamera::Buffer& BaseCamera::buffer() const {
	return hostBuffer;
}

void BaseCamera::setTransformation(const math::mat4& transformation) {
	buffer(true).view = transpose(inverse(transformation));
}

void BaseCamera::setViewport(float width, float height) {
	buffer(true).viewport = math::vec2{width, height};
}

}