#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <cudaRayTracing/hitable/triangle.h>

#include <vector>

namespace cuda::rayTracing {

struct Primitive{
    Triangle shape;
    box bbox;

    box getBox() const { return bbox;}
};

void sortByBox(std::vector<const Primitive*>::iterator begin, std::vector<const Primitive*>::iterator end, const box& bbox);
std::vector<Triangle> extractTriangles(const std::vector<const Primitive*>& storage);

}
#endif // PRIMITIVE_H
