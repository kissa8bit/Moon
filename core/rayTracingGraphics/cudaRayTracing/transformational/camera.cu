#include "camera.h"

namespace cuda::rayTracing {

__host__ __device__ void Camera::update(){
    horizontal = aspect * vec4f::getHorizontal(viewRay.getDirection());
    vertical = vec4f::getVertical(viewRay.getDirection());
}

__host__ __device__ Camera::Camera(
    const ray viewRay,
    float aspect,
    float matrixScale,
    float matrixOffset,
    float focus,
    float apertura) :
    viewRay(viewRay),
    aspect(aspect),
    matrixScale(matrixScale),
    matrixOffset(matrixOffset),
    focus(focus),
    apertura(apertura)
{
    update();
}

__device__ ray Camera::getPixelRay(float u, float v, curandState* randState) const {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u + apertura * float(curand_uniform(randState));
    v = matrixScale * t * v + apertura * float(curand_uniform(randState));
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

__host__ __device__ ray Camera::getPixelRay(float u, float v) const {
    const float t = focus / (matrixOffset - focus);
    u = matrixScale * t * u;
    v = matrixScale * t * v;
    return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
}

}
