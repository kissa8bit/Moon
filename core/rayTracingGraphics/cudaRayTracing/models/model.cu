#include "models/model.h"
#include "hitable/triangle.h"

namespace cuda::rayTracing {

Model::~Model(){}

Model::Model(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer, const std::vector<cudaTextureObject_t>& textures)
    : vertexBuffer(Buffer<Vertex>(vertexBuffer.size(), vertexBuffer.data()))
{
    for (size_t index = 0; index < indexBuffer.size(); index += 3) {
        cudaTextureObject_t tex = textures.empty() ? 0 : textures.front();
        Triangle tr(indexBuffer[index + 0], indexBuffer[index + 1], indexBuffer[index + 2], vertexBuffer.data());
        primitives.push_back({
            make_devicep<Hitable>(
                Triangle(
                    indexBuffer[index + 0],
                    indexBuffer[index + 1],
                    indexBuffer[index + 2],
                    this->vertexBuffer.get(),
                    tex
                )
            ),
            tr.getBox()
        });
    }
}

Model::Model(std::vector<Primitive>&& primitives)
    : primitives(std::move(primitives)) {}

Model::Model(Primitive&& primitive){
    primitives.emplace_back(std::move(primitive));
}

Model::Model(Model&& m) : vertexBuffer(std::move(m.vertexBuffer)), primitives(std::move(m.primitives))
{}

Model& Model::operator=(Model&& m)
{
    primitives.clear();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    vertexBuffer = std::move(m.vertexBuffer);
    primitives = std::move(m.primitives);
    return *this;
}

}
