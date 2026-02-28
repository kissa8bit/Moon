#ifndef MOON_TRANSFORMATIONALS_LIGHTS_H
#define MOON_TRANSFORMATIONALS_LIGHTS_H

#include "transformational.h"

#include <interfaces/light.h>

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Light : public Transformational {
    DEFAULT_TRANSFORMATIONAL()

protected:
    Light() = default;

public:
    virtual ~Light() override = default;

    Light(const Light&) = delete;
    Light& operator=(const Light&) = delete;

    Light(Light&&) = default;
    Light& operator=(Light&&) = default;

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Light)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Light)

    virtual interfaces::Light* light() = 0;
};

} // moon::transformational

#endif // MOON_TRANSFORMATIONALS_LIGHTS_H
