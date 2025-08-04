#ifndef MOON_ENTITIES_BASE_CAMERA_H
#define MOON_ENTITIES_BASE_CAMERA_H

#include <transformationals/cameras.h>

#include <math/linearAlgebra.h>

namespace moon::entities {

class BaseCamera : public transformational::Camera
{
public:
    BaseCamera(const float& angle, const float& aspect, const float& n, const float& f = std::numeric_limits<float>::max());

    BaseCamera(const BaseCamera&) = delete;
    BaseCamera& operator=(const BaseCamera&) = delete;

    BaseCamera(BaseCamera&&) = default;
    BaseCamera& operator=(BaseCamera&&) = default;

    BaseCamera& setProjMatrix(const math::mat4& proj);
    math::mat4 getProjMatrix() const;
    math::mat4 getViewMatrix() const;
};

} // moon::entities

#endif // MOON_ENTITIES_BASE_CAMERA_H