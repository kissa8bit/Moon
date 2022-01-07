#ifndef HITABLECONTAINER_H
#define HITABLECONTAINER_H

#include <vector>
#include "hitable.h"

namespace cuda::rayTracing {

class HitableContainer {
protected:
    size_t container_size{0};

public:
    template <typename type>
    class BaseIterator{
    protected:
        type* ptr{nullptr};

    public:
        typedef type value_type;

        __host__ __device__ BaseIterator() {};
        __host__ __device__ BaseIterator(type* ptr) : ptr(ptr) {}

        __host__ __device__ const Hitable*& operator*() const { return (*ptr)(); }
        __host__ __device__ const Hitable** operator->() { return ptr(); }
        __host__ __device__ BaseIterator& operator++() { ptr = ptr->get_next(); return *this; }
        __host__ __device__ BaseIterator operator++(int) { BaseIterator tmp = *this; ++(*this); return tmp; }
        __host__ __device__ friend bool operator== (const BaseIterator& a, const BaseIterator& b) { return a.ptr == b.ptr; };
        __host__ __device__ friend bool operator!= (const BaseIterator& a, const BaseIterator& b) { return a.ptr != b.ptr; };
        __host__ __device__ friend BaseIterator operator+ (BaseIterator it, size_t s) {
            for(; s > 0; s--) it.ptr = it.ptr->get_next();
            return it;
        };
    };

    __host__ __device__ virtual ~HitableContainer(){}
    __host__ __device__ virtual bool hit(const ray& r, HitCoords& coord) const = 0;

    __host__ __device__ virtual void add(const Hitable*const* objects, size_t size = 1) = 0;

    __host__ __device__ virtual const Hitable*& operator[](uint32_t i) const = 0;
    __host__ __device__ virtual size_t size() const { return container_size; }

    static void destroy(HitableContainer* dpointer);
    static void add(HitableContainer* dpointer, const std::vector<const Hitable*>& objects);
};

}
#endif // HITABLECONTAINER_H
