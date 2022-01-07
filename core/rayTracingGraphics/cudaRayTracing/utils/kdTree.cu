#include "utils/kdTree.h"
#include "utils/operations.h"
#include "utils/buffer.h"

namespace cuda::rayTracing {

__global__ void setRootKernel(HitableKDTree* tree, HitableKDTree::KDNodeType* nodesBuffer)
{
    tree->setRoot(&nodesBuffer[0]);
}

__global__ void createTreeKernel(HitableKDTree* tree, HitableKDTree::KDNodeType* nodesBuffer, NodeDescriptor* nodeDescriptors)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    tree->makeTree(nodesBuffer, nodeDescriptors[i]);
}

void makeTree(HitableKDTree* container, NodeDescriptor* nodeDescriptors, size_t size){
    Buffer<HitableKDTree::KDNodeType> nodesBuffer(size);
    createTreeKernel<<<size,1>>>(container, nodesBuffer.get(), nodeDescriptors);
    setRootKernel<<<1,1>>>(container, nodesBuffer.release());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void createKernel(HitableKDTree* p) {
    p = new (p) HitableKDTree();
}

void HitableKDTree::create(HitableKDTree* dpointer, const HitableKDTree& host){
    createKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(HitableKDTree* p) {
    p->~HitableKDTree();
}

void HitableKDTree::destroy(HitableKDTree* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

}
