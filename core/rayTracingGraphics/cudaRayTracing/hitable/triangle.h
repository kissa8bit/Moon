#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "hitable.h"

namespace cuda::rayTracing {

// Shared per-model data: stored once on device, referenced by all triangles.
struct MeshData {
    const Vertex*          vertexBuffer{ nullptr };
    cudaTextureObject_t    texture{ 0 };
};

struct Triangle {
    uint32_t          index[3]{ 0, 0, 0 };
    const MeshData*   mesh{ nullptr };

    __host__ __device__ Triangle() = default;

    __host__ __device__ Triangle(uint32_t i0, uint32_t i1, uint32_t i2, const MeshData* mesh)
        : index{i0, i1, i2}, mesh(mesh) {}

    __host__ __device__ bool hit(const ray& r, HitCoords& coord) const {
        const Vertex* vb = mesh->vertexBuffer;
        const vec4f a = vb[index[1]].point - r.getOrigin();
        const vec4f b = vb[index[1]].point - vb[index[2]].point;
        const vec4f c = vb[index[1]].point - vb[index[0]].point;
        const vec4f& d = r.getDirection();

        float det = det3(d, b, c);
        if (det == 0.0f) return false;
        det = 1.0f / det;

        const float t = det3(a, b, c) * det;
        if (t > coord.tmax || t < coord.tmin) return false;

        const float u = det3(d, a, c) * det;
        const float v = det3(d, b, a) * det;
        if (u < 0.0f || v < 0.0f || u + v > 1.0f) return false;

        coord = { coord.tmin, t, u, v };
        return true;
    }

#ifdef __CUDACC__
    __device__ void calcHitRecord(const ray& r, const HitCoords& coord, HitRecord& rec) const {
        const float s              = 1.0f - coord.u - coord.v;
        const Vertex* vb           = mesh->vertexBuffer;
        const cudaTextureObject_t texture = mesh->texture;
        const uint32_t* idx        = index;

        rec.vertex.point = r.point(coord.tmax);
        {
            const vec4f localNormal = normal(coord.v * vb[idx[0]].normal + coord.u * vb[idx[2]].normal + s * vb[idx[1]].normal);
            rec.vertex.normal = coord.toWorld ? normal((*coord.toWorld) * localNormal) : localNormal;
        }
        rec.vertex.u = coord.v * vb[idx[0]].u + coord.u * vb[idx[2]].u + s * vb[idx[1]].u;
        rec.vertex.v = coord.v * vb[idx[0]].v + coord.u * vb[idx[2]].v + s * vb[idx[1]].v;

        if (texture) {
            const uchar4 color = tex2D<uchar4>(texture, rec.vertex.u, 1.0f - rec.vertex.v);
            rec.vertex.color   = vec4f(color.x, color.y, color.z, color.w) / 255.0f;
        } else {
            rec.vertex.color = coord.v * vb[idx[0]].color + coord.u * vb[idx[2]].color + s * vb[idx[1]].color;
        }

        rec.vertex.props = {
            coord.v * vb[idx[0]].props.refractiveIndex  + coord.u * vb[idx[2]].props.refractiveIndex  + s * vb[idx[1]].props.refractiveIndex,
            coord.v * vb[idx[0]].props.refractProb      + coord.u * vb[idx[2]].props.refractProb      + s * vb[idx[1]].props.refractProb,
            coord.v * vb[idx[0]].props.fuzz             + coord.u * vb[idx[2]].props.fuzz             + s * vb[idx[1]].props.fuzz,
            coord.v * vb[idx[0]].props.angle            + coord.u * vb[idx[2]].props.angle            + s * vb[idx[1]].props.angle,
            coord.v * vb[idx[0]].props.emissionFactor   + coord.u * vb[idx[2]].props.emissionFactor   + s * vb[idx[1]].props.emissionFactor,
            coord.v * vb[idx[0]].props.absorptionFactor + coord.u * vb[idx[2]].props.absorptionFactor + s * vb[idx[1]].props.absorptionFactor
        };
    }
#endif
};

}
#endif // TRIANGLE_H
