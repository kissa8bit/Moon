#ifndef OBJECTH
#define OBJECTH

#include "models/model.h"

namespace cuda::rayTracing {

struct Object{
    Model* model{nullptr};
    mat4f transform = mat4f::identity();

    Object(Model* model, const mat4f& transform = mat4f::identity()) : model(model), transform(transform) {}
    ~Object(){ if(model) delete model;}
};

}
#endif // !OBJECTH
