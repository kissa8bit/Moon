#ifndef BOX_H
#define BOX_H
#include "ray.h"

namespace cuda::rayTracing {

struct box{
    vec4f min{std::numeric_limits<float>::max()};
    vec4f max{std::numeric_limits<float>::lowest()};

    __host__ __device__ float surfaceArea() const {
        const float dx = max[0] - min[0];
        const float dy = max[1] - min[1];
        const float dz = max[2] - min[2];
        return 2.0f * (dx * dy + dz * dy + dx * dz);
    }

    __host__ __device__ bool intersect(const ray &r) const {
        float dx = 1.0f / r.getDirection()[0];
        float dy = 1.0f / r.getDirection()[1];
        float dz = 1.0f / r.getDirection()[2];

        float t1 = (min[0] - r.getOrigin()[0]) * dx;
        float t2 = (max[0] - r.getOrigin()[0]) * dx;
        float t3 = (min[1] - r.getOrigin()[1]) * dy;
        float t4 = (max[1] - r.getOrigin()[1]) * dy;
        float t5 = (min[2] - r.getOrigin()[2]) * dz;
        float t6 = (max[2] - r.getOrigin()[2]) * dz;

        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
        // if tmin > tmax, ray doesn't intersect AABB
        return tmax >= 0 && tmin <= tmax;
    }
};

struct cbox : public box{
    vec4f color{0.0f, 0.0f, 0.0f, 0.0f};
    __host__ __device__ cbox() {}
    __host__ __device__ cbox(const box& b) : box(b) {}
    __host__ __device__ cbox(const box& b, const vec4f& color) : box(b), color(color) {}
};

}

#endif // BOX_H
