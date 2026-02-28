#ifndef MOON_TRANSFORMATIONALS_OBJECTS_H
#define MOON_TRANSFORMATIONALS_OBJECTS_H

#include "transformational.h"

#include <interfaces/object.h>

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Object : public Transformational
{
    DEFAULT_TRANSFORMATIONAL()

protected:
    Object() = default;

public:
    virtual ~Object() override = default;

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    Object(Object&&) = default;
    Object& operator=(Object&&) = default;

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Object)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    virtual interfaces::Object* object() = 0;
};

}
#endif // MOON_TRANSFORMATIONALS_OBJECTS_H
