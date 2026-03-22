#include "object.h"

#include <utils/operations.h>

namespace moon::interfaces {

Object::Object(ObjectMask objectMask, Model* model, const Range& instanceRange)
    : mask(objectMask), pModel(model), instance({ instanceRange })
{}

Model* Object::model() {
    return pModel;
}

const Model* Object::model() const {
    return pModel;
}

uint32_t Object::getInstanceNumber(utils::ResourceIndex resourceIndex) const {
    return instance.range.count > 1 ? instance.range.first + resourceIndex.get() : instance.range.first;
}

uint32_t Object::instanceCount() const {
    return instance.range.count;
}

ObjectMask& Object::objectMask() {
    return mask;
}

const ObjectMask& Object::objectMask() const {
    return mask;
}

Range& Object::primitiveRange() { return primitive.range;}

bool Object::comparePrimitive(uint32_t primitiveIndex) const {
    return !(primitiveIndex < primitive.range.first) && (primitiveIndex < primitive.range.last());
}

const VkDescriptorSet& Object::getDescriptorSet(utils::ResourceIndex resourceIndex) const {
    return descriptors.at(resourceIndex.get());
}

}
