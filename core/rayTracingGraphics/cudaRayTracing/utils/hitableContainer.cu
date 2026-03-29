#include "hitableContainer.h"

#include "operations.h"
#include "buffer.h"

namespace cuda::rayTracing {

__global__ void addKernel(HitableContainer* container, const Triangle* objects, size_t size) {
    container->add(objects, size);
}

void HitableContainer::add(HitableContainer* dpointer, const std::vector<Triangle>& objects) {
    Buffer<Triangle> devBuffer(objects.size(), objects.data());
    addKernel<<<1,1>>>(dpointer, devBuffer.get(), devBuffer.getSize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableContainer* p) {
    p->~HitableContainer();
}

void HitableContainer::destroy(HitableContainer* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
