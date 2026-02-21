#include "object.h"

#include <utils/operations.h>

namespace moon::interfaces {

Object::Object(ObjectMask objectMask, Model* model, const Range& instanceRange)
    : mask(objectMask), pModel(model), instance({ instanceRange })
{}

Model* Object::model() {
    return pModel;
}

uint32_t Object::getInstanceNumber(uint32_t imageNumber) const {
    return instance.range.count > 1 ? instance.range.first + imageNumber : instance.range.first;
}

ObjectMask& Object::objectMask() {
    return mask;
}

Range& Object::primitiveRange() { return primitive.range;}

bool Object::comparePrimitive(uint32_t primitiveIndex) const {
    return !(primitiveIndex < primitive.range.first) && (primitiveIndex < primitive.range.last());
}

const VkDescriptorSet& Object::getDescriptorSet(uint32_t i) const {
    return descriptors.at(i);
}

}
