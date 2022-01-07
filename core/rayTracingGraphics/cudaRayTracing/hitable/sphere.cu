#include "sphere.h"
#include "operations.h"

namespace cuda::rayTracing {

__host__ __device__ Sphere::Sphere(const vec4f& cen, float r, const vec4f& color, const Properties& props) : center(cen), radius(r), color(color), props(props) {}

__host__ __device__ Sphere::Sphere(const vec4f& cen, float r, const vec4f& color) : center(cen), radius(r), color(color) {}

__host__ __device__ bool Sphere::hit(const ray& r, HitCoords& coord) const {
    vec4f oc = r.getOrigin() - center;
    float a = 1.0f / r.getDirection().length2();
    float b = - dot(oc, r.getDirection()) * a;
    float c = oc.length2() - radius * radius * a;
    float discriminant = b * b - c;

    if (discriminant < 0) {
        return false;
    }

    discriminant = sqrt(discriminant);
    float temp = b - discriminant;
    bool result = (temp < coord.tmax && temp > coord.tmin);
    if (!result) {
        temp = b + discriminant;
        result = (temp < coord.tmax && temp > coord.tmin);
    }
    if (result) {
        coord.tmax = temp;
    }
    return result;
}

__device__ void Sphere::calcHitRecord(const ray& r, const HitCoords& coord, HitRecord& rec) const {
    rec.vertex.point = r.point(coord.tmax);
    rec.vertex.normal = (rec.vertex.point - center) / radius;
    rec.vertex.color = color;
    rec.vertex.props = props;
}

__global__ void createKernel(Sphere* sph, vec4f cen, float r, vec4f color, const Properties props) {
    sph = new (sph) Sphere(cen, r, color, props);
}

void Sphere::create(Sphere* dpointer, const Sphere& host){
    createKernel<<<1,1>>>(dpointer, host.center, host.radius, host.color, host.props);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void destroyKernel(Sphere* p) {
    p->~Sphere();
}

void Sphere::destroy(Sphere* dpointer){
    destroyKernel<<<1,1>>>(dpointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ __device__ box Sphere::getBox() const {
    box bbox;
    bbox.min = center - vec4f(radius, radius, radius, 0.0f);
    bbox.max = center + vec4f(radius, radius, radius, 0.0f);
    return bbox;
}

}
