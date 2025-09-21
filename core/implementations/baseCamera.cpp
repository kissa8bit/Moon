#include "baseCamera.h"

namespace moon::implementations {

BaseCamera::BaseCamera()
    : uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void BaseCamera::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseCamera::create(const utils::PhysicalDevice& device, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
}

utils::Buffers& BaseCamera::buffers() {
    return uniformBuffer.device;
}

BaseCamera::Buffer& BaseCamera::buffer(bool update) {
	if (update) {
		utils::vkDefault::raiseFlags(uniformBuffer.device);
	}
	return hostBuffer;
}

void BaseCamera::setTransformation(const math::mat4& transformation) {
	buffer(true).view = transpose(inverse(transformation));
}

}