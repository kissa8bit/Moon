#include "material.h"

namespace cuda::rayTracing {

__device__ vec4f scatter(const ray& r, const vec4f& norm, const Properties& props, curandState* randState)
{
    vec4f scattered = vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    if(props.emissionFactor >= 0.98f) { return scattered;}

    const vec4f& d = r.getDirection();

    if (curand_uniform(randState) <= props.refractProb) {
        const vec4f n = (dot(d, norm) <= 0.0f) ? norm : - norm;
        const float eta = (dot(d, norm) <= 0.0f) ? (1.0f / props.refractiveIndex) : props.refractiveIndex;

        float cosPhi = dot(d, n);
        float sinTheta = eta * std::sqrt(1.0f - cosPhi * cosPhi);
        if (std::abs(sinTheta) <= 1.0f) {
            float cosTheta = std::sqrt(1.0f - sinTheta * sinTheta);
            vec4f tau = normal(d - dot(d, n) * n);
            scattered = sinTheta * tau - cosTheta * n;
        }
        if (scattered.length2() != 0.0f) { return scattered;}
    }

    if (props.fuzz > 0.0f) {
        vec4f reflect = normal(d + 2.0f * std::abs(dot(d, norm)) * norm);
        scattered = reflect + props.fuzz * random_in_unit_sphere(reflect, props.angle, randState);
        scattered = (dot(norm, scattered) > 0.0f ? 1.0f : 0.0f) * scattered;
    } else {
        scattered = random_in_unit_sphere(norm, props.angle, randState);
        scattered = (dot(norm, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
    }
    return scattered;
}

}
