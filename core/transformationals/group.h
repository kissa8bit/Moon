#ifndef GROUP_H
#define GROUP_H

#include <unordered_set>

#include "transformational.h"

#include "linearAlgebra.h"

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

}
#endif // GROUP_H
