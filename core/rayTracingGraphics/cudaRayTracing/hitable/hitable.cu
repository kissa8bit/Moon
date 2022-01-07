#include "hitable.h"
#include "operations.h"

namespace cuda::rayTracing {

__global__ void destroyKernel(Hitable* p) {
    p->~Hitable();
}

void Hitable::destroy(Hitable* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
