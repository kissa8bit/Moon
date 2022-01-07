#ifndef HITABLEARRAY_H
#define HITABLEARRAY_H

#include "hitableContainer.h"

namespace cuda::rayTracing {

class HitableArray : public HitableContainer {
public:
    struct Pointer{
        const Hitable* p{nullptr};
        __host__ __device__ Pointer* get_next() { return this + 1; }
        __host__ __device__ const Hitable*& operator()() { return p; }
    };

    using iterator = BaseIterator<Pointer>;

    __host__ __device__ HitableArray(){};
    __host__ __device__ ~HitableArray();

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) const override;

    __host__ __device__ void add(const Hitable*const* objects, size_t size = 1) override;

    __host__ __device__ const Hitable*& operator[](uint32_t i) const override;

    __host__ __device__ iterator begin() {return iterator(&array[0]); }
    __host__ __device__ iterator end() {return iterator(&array[container_size]); }

    __host__ __device__ iterator begin() const {return iterator(&array[0]); }
    __host__ __device__ iterator end() const {return iterator(&array[container_size]); }

    static void create(HitableArray* dpointer, const HitableArray& host);
    static void destroy(HitableArray* dpointer);

private:
    Pointer* array{ nullptr };
};

}
#endif // HITABLEARRAY_H
