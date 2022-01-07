#ifndef MOON_TRANSFORMATIONALS_OBJECTS_H
#define MOON_TRANSFORMATIONALS_OBJECTS_H

#include <vulkan.h>

#include <filesystem>
#include <optional>

#include "transformational.h"

#include <utils/texture.h>
#include <utils/buffer.h>

#include <interfaces/object.h>

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Object : public Transformational
{
protected:
    std::unique_ptr<interfaces::Object> pObject;

    DEFAULT_TRANSFORMATIONAL()

    Object() = default;

public:
    virtual ~Object() override = default;

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Object)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    operator interfaces::Object* () const;
};

}
#endif // MOON_TRANSFORMATIONALS_OBJECTS_H
