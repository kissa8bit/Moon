#include "kdTree.h"
#include "operations.h"
#include "buffer.h"

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

__global__ void destroyKernel(HitableKDTree* p, HitableKDTree::KDNodeType** rootOut) {
    // Null the root before calling the destructor so ~HitableKDTree() skips
    // `delete[] root` — the node buffer was allocated via cudaMalloc (through
    // Buffer<>) and must be freed with cudaFree from the host, not delete[].
    *rootOut = p->releaseRoot();
    p->~HitableKDTree();
}

void HitableKDTree::destroy(HitableKDTree* dpointer){
    Buffer<KDNodeType*> rootPtrBuf(1);
    destroyKernel<<<1,1>>>(dpointer, rootPtrBuf.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    KDNodeType* rootPtr = nullptr;
    checkCudaErrors(cudaMemcpy(&rootPtr, rootPtrBuf.get(), sizeof(KDNodeType*), cudaMemcpyDeviceToHost));
    if (rootPtr) checkCudaErrors(cudaFree(rootPtr));
}

}
