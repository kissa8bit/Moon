#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "hitable/hitable.h"
#include "utils/devicep.h"

#include <vector>

namespace cuda::rayTracing {

struct Primitive{
    Devicep<Hitable> hit;
    box bbox;

    box getBox() const { return bbox;}
};

void sortByBox(std::vector<const Primitive*>::iterator begin, std::vector<const Primitive*>::iterator end, const box& bbox);
std::vector<const Hitable*> extractHitables(const std::vector<const Primitive*>& storage);

}
#endif // PRIMITIVE_H
