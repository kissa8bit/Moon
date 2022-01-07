#ifndef MOON_TRANSFORMATIONALS_CAMERAS_H
#define MOON_TRANSFORMATIONALS_CAMERAS_H

#include <vulkan.h>

#include <interfaces/camera.h>

#include <math/linearAlgebra.h>

#include "transformational.h"

namespace moon::transformational {

class Camera : public Transformational
{
protected:
    std::unique_ptr<interfaces::Camera> pCamera;

    DEFAULT_TRANSFORMATIONAL()

    Camera() = default;

public:
    virtual ~Camera() override = default;

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    Camera(Camera&&) = default;
    Camera& operator=(Camera&&) = default;

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Camera)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Camera)

    operator interfaces::Camera* () const;
};

}

#endif // MOON_TRANSFORMATIONALS_CAMERAS_H
