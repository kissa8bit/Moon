#include "hitableArray.h"

#include "operations.h"

namespace cuda::rayTracing {

__host__ __device__ HitableArray::~HitableArray() {
    if (array) {
        delete[] array;
    }
    container_size = 0;
}

__host__ __device__ bool HitableArray::hit(const ray& r, HitCoords& coord) const {
    for(iterator it = begin(); it != end(); it++){
        if ((*it)->hit(r, coord)) {
            coord.obj = *it;
        }
    }
    return coord.obj;
}

__host__ __device__ void HitableArray::add(const Hitable*const* object, size_t size) {
    Pointer* newArray = new Pointer[container_size + size];
    for(size_t i = 0; i < container_size; i++){
        newArray[i] = array[i];
    }
    for(size_t i = 0; i < size; i++){
        newArray[container_size + i].p = object[i];
    }
    if (array) {
        delete[] array;
    }
    array = newArray;
    container_size += size;
}

__host__ __device__ const Hitable*& HitableArray::operator[](uint32_t i) const {
    return array[i].p;
}

__global__ void createKernel(HitableArray* p) {
    p = new (p) HitableArray();
}

void HitableArray::create(HitableArray* dpointer, const HitableArray& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableArray* p) {
    p->~HitableArray();
}

void HitableArray::destroy(HitableArray* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
