#include "cameras.h"

#include "operations.h"
#include "device.h"

#include <cstring>

namespace moon::interfaces {

BaseCamera::BaseCamera(void* hostData, size_t hostDataSize)
    : uniformBuffer(hostData, hostDataSize) {}

void BaseCamera::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseCamera::create(const utils::PhysicalDevice& device, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
}

utils::Buffers& BaseCamera::buffers() {
    return uniformBuffer.device;
}

}

namespace moon::transformational {

Camera::Camera() {
    pCamera = std::make_unique<interfaces::BaseCamera>(&buffer, sizeof(buffer));
}

Camera::Camera(const float& angle, const float& aspect, const float& near, const float& far) : Camera() {
    setProjMatrix(math::perspective(math::radians(angle), aspect, near, far));
}

Camera& Camera::update() {
    math::mat4 transformMatrix = convert(convert(m_rotation, m_translation));
    buffer.view = transpose(inverse(math::mat4(m_globalTransformation * transformMatrix)));
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

Camera& Camera::setProjMatrix(const math::mat4& proj) {
    buffer.proj = transpose(proj);
    utils::raiseFlags(pCamera->buffers());
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Camera)

math::mat4 Camera::getProjMatrix()  const { return transpose(buffer.proj);}
math::mat4 Camera::getViewMatrix()  const { return transpose(buffer.view);}

Camera::operator interfaces::Camera* () const {
    return pCamera.get();
}

}
