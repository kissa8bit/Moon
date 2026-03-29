#ifndef OBJECTINSTANCE_H
#define OBJECTINSTANCE_H

#include <cudaRayTracing/math/mat4.h>
#include <cudaRayTracing/math/box.h>
#include <cudaRayTracing/hitable/hitable.h>
#include <cudaRayTracing/utils/kdTree.h>

namespace cuda::rayTracing {

// One instance of an object in the scene: stores the per-object BLAS (in local space)
// plus the world-space transform. Used as elements of the flat TLAS.
struct ObjectInstance {
    mat4f toWorld = mat4f::identity();
    mat4f toLocal = mat4f::identity();
    box worldBbox;
    HitableKDTree* blas{nullptr};

    __host__ __device__ box getBox() const { return worldBbox; }

    // Transform the world ray to local space and traverse the per-object BLAS.
    // Because ray() normalises its direction, the t parametrisation differs:
    //   t_local = t_world * scale,  where scale = |toLocal * worldDir|
    // coord.tmax is maintained in world-t space; we scale in/out at the boundary.
    // coord.toWorld is set so Triangle::calcHitRecord can rotate normals back.
    __device__ bool hit(const ray& worldRay, HitCoords& coord) const {
        const vec4f localDirRaw = toLocal * worldRay.getDirection();  // un-normalised
        const float scale       = localDirRaw.length();               // t scale factor

        const ray   localRay(toLocal * worldRay.getOrigin(), localDirRaw);  // ctor normalises dir

        HitCoords localCoord;
        localCoord.tmin = coord.tmin * scale;
        localCoord.tmax = coord.tmax * scale;

        if (!blas->hit(localRay, localCoord)) return false;

        coord.tmax    = localCoord.tmax / scale;   // convert back to world-t
        coord.u       = localCoord.u;
        coord.v       = localCoord.v;
        coord.obj     = localCoord.obj;
        coord.toWorld = &toWorld;
        return true;
    }
};

}
#endif // OBJECTINSTANCE_H
