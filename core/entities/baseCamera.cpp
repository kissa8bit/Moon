#include "baseCamera.h"

#include <implementations/baseCamera.h>

namespace moon::entities {

BaseCamera::BaseCamera(const float& angle, const float& aspect, const float& near, const float& far)
    : transformational::Camera()
{
    pCamera = std::make_unique<implementations::BaseCamera>();

    setProjMatrix(math::perspective(math::radians(angle), aspect, near, far));
}

math::mat4 BaseCamera::getProjMatrix() const {
    if (auto pBaseCamera = static_cast<implementations::BaseCamera*>(pCamera.get()); pBaseCamera) {
        return transpose(pBaseCamera->buffer(true).proj);
    }
    return math::mat4::identity();
}

math::mat4 BaseCamera::getViewMatrix() const {
    if (auto pBaseCamera = static_cast<implementations::BaseCamera*>(pCamera.get()); pBaseCamera) {
        return transpose(pBaseCamera->buffer(true).view);
    }
    return math::mat4::identity();
}

BaseCamera& BaseCamera::setProjMatrix(const math::mat4& proj) {
    if (auto pBaseCamera = static_cast<implementations::BaseCamera*>(pCamera.get()); pBaseCamera) {
        pBaseCamera->buffer(true).proj = transpose(proj);
    }
    return *this;
}

} // moon::entities