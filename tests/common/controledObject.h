#ifndef CONTROLED_OBJ_H
#define CONTROLED_OBJ_H

#include "vector.h"
#include "objects.h"

namespace moon::tests {

struct Outlighting {
    bool enable{ true };
    moon::math::Vector<float, 4> color{ 1.0f };
};

struct ControledObject {
    moon::transformational::Object* ptr{ nullptr };
    std::string name{ "none" };
    Outlighting outlighting;

    operator moon::transformational::Object* () { return ptr; }
    operator moon::transformational::Object& () { return *ptr; }
    moon::transformational::Object* operator->() { return ptr; }
};

}

#endif // CONTROLED_OBJ_H