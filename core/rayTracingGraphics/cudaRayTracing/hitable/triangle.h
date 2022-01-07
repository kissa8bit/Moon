#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"

namespace cuda::rayTracing {

class Triangle : public Hitable {
private:
    size_t index[3];
    const Vertex* vertexBuffer{ nullptr };
    cudaTextureObject_t texture{ 0 };

public:
    __host__ __device__ Triangle() {}
    __host__ __device__ ~Triangle() {}

    __host__ __device__ Triangle(const size_t& i0, const size_t& i1, const size_t& i2, const Vertex* vertexBuffer, cudaTextureObject_t texture = 0);
    __host__ __device__ bool hit(const ray& r, HitCoords& coords) const override;
    __device__ void calcHitRecord(const ray& r, const HitCoords& coords, HitRecord& rec) const override;

    static void create(Triangle* dpointer, const Triangle& host);
    static void destroy(Triangle* dpointer);
    __host__ __device__ box getBox() const override;
};

}

#endif
