#ifndef MATERIALH
#define MATERIALH

#include "math/ray.h"

namespace cuda::rayTracing {

struct Properties {
    float refractiveIndex{ 1.0f };
    float refractProb{ 0.0f };
    float fuzz{ 0.0f };
    float angle{ 0.0f };
    float emissionFactor{ 0.0f };
    float absorptionFactor{ 1.0f };
};

__device__ vec4f scatter(const ray& r, const vec4f& norm, const Properties& props, curandState* randState);

}

#endif
