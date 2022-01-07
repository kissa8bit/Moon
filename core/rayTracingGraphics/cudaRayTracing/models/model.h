#ifndef MODELH
#define MODELH

#include <cudaRayTracing/utils/buffer.h>
#include <cudaRayTracing/utils/primitive.h>
#include <cudaRayTracing/math/mat4.h>

namespace cuda::rayTracing {

class Model {
public:
    Buffer<Vertex> vertexBuffer;
    std::vector<Primitive> primitives;
    std::vector<cudaTextureObject_t> textures;

    Model(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer, const std::vector<cudaTextureObject_t>& textures = {});
    Model(std::vector<Primitive>&& primitives);
    Model(Primitive&& primitive);
    Model() = default;
    Model(Model&& m);
    Model& operator=(Model&& m);

    virtual ~Model();

    virtual void load(const mat4f& transform) {}
};

}
#endif
