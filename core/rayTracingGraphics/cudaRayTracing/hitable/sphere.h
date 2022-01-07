#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

namespace cuda::rayTracing {

class Sphere : public Hitable {
private:
    vec4f center{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4f color{ 0.0f,0.0f, 0.0f, 0.0f };
    float radius{ 0.0f };
    Properties props;

public:
    __host__ __device__ Sphere() {}
    __host__ __device__ ~Sphere() {}

    __host__ __device__ Sphere(const vec4f& cen, float r, const vec4f& color, const Properties& props);
    __host__ __device__ Sphere(const vec4f& cen, float r, const vec4f& color);
    __host__ __device__ bool hit(const ray& r, HitCoords& coords) const override;
    __device__ void calcHitRecord(const ray& r, const HitCoords& coords, HitRecord& rec) const override;

    static void create(Sphere* dpointer, const Sphere& host);
    static void destroy(Sphere* dpointer);
    __host__ __device__ box getBox() const override;
};

}

#endif
