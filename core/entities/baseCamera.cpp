#include "baseCamera.h"

#include <math/linearAlgebra.h>

namespace moon::entities {

BaseCamera::BaseCamera(const float& angle, const float& aspect, const float& near, const float& far)
    : transformational::Camera()
{
    setProjMatrix(math::perspective(math::radians(angle), aspect, near, far));
}

math::mat4 BaseCamera::getProjMatrix() const {
    return math::transpose(m_camera.buffer().proj);
}

math::mat4 BaseCamera::getViewMatrix() const {
    return math::transpose(m_camera.buffer().view);
}

BaseCamera& BaseCamera::setProjMatrix(const math::mat4& proj) {
    m_camera.buffer(true).proj = math::transpose(proj);
    return *this;
}

} // moon::entities
