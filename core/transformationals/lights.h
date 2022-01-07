#ifndef MOON_TRANSFORMATIONALS_LIGHTS_H
#define MOON_TRANSFORMATIONALS_LIGHTS_H

#include <filesystem>
#include <memory>

#include "transformational.h"

#include <interfaces/light.h>

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Light : public Transformational {
protected:
    std::unique_ptr<interfaces::Light> pLight;

    DEFAULT_TRANSFORMATIONAL()

    Light() = default;

public:
    virtual ~Light() override = default;

    Light(const Light&) = delete;
    Light& operator=(const Light&) = delete;

    Light(Light&& other) { std::swap(pLight, other.pLight); };
    Light& operator=(Light&& other) { std::swap(pLight, other.pLight); return *this;};

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Light)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Light)

    operator interfaces::Light* () const;
};

} // moon::transformational

#endif // MOON_TRANSFORMATIONALS_LIGHTS_H
