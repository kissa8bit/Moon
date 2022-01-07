#ifndef MOON_TRANSFORMATIONALS_GROUP_H
#define MOON_TRANSFORMATIONALS_GROUP_H

#include <unordered_set>

#include "transformational.h"

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Group : public Transformational
{
private:
    std::unordered_set<Transformational*> objects;
    DEFAULT_TRANSFORMATIONAL()

public:
    virtual ~Group() = default;

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Group)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    bool add(Transformational* object);
    bool remove(Transformational* object);
    bool find(Transformational* object);
};

} // moon::transformational

#endif // MOON_TRANSFORMATIONALS_GROUP_H
