#include "model.h"

#include <cudaRayTracing/hitable/triangle.h>

namespace cuda::rayTracing {

Model::~Model(){}

Model::Model(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer, const std::vector<cudaTextureObject_t>& textures)
    : vertexBuffer(Buffer<Vertex>(vertexBuffer.size(), vertexBuffer.data()))
{
    // Upload shared per-model data once; all triangles reference this device pointer.
    const MeshData hostMesh{ this->vertexBuffer.get(), textures.empty() ? 0 : textures.front() };
    meshData = Buffer<MeshData>(1, &hostMesh);
    const MeshData* devMesh = meshData.get();

    for (size_t i = 0; i < indexBuffer.size(); i += 3) {
        const uint32_t i0 = indexBuffer[i], i1 = indexBuffer[i + 1], i2 = indexBuffer[i + 2];
        box bbox;
        bbox.min = min(vertexBuffer[i0].point, min(vertexBuffer[i1].point, vertexBuffer[i2].point));
        bbox.max = max(vertexBuffer[i0].point, max(vertexBuffer[i1].point, vertexBuffer[i2].point));
        primitives.push_back({ Triangle(i0, i1, i2, devMesh), bbox });
    }
}

Model::Model(std::vector<Primitive>&& primitives)
    : primitives(std::move(primitives)) {}

Model::Model(Primitive&& primitive){
    primitives.emplace_back(std::move(primitive));
}

Model::Model(Model&& m) : vertexBuffer(std::move(m.vertexBuffer)), meshData(std::move(m.meshData)), primitives(std::move(m.primitives))
{}

Model& Model::operator=(Model&& m)
{
    primitives.clear();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    vertexBuffer = std::move(m.vertexBuffer);
    meshData = std::move(m.meshData);
    primitives = std::move(m.primitives);
    return *this;
}

}
