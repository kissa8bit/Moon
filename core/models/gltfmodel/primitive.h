#ifndef GLTFMODEL_PRIMITIVE_H
#define GLTFMODEL_PRIMITIVE_H

#include "model.h"

namespace moon::models {

struct Primitive {
    uint32_t firstIndex{ 0 };
    uint32_t indexCount{ 0 };
    uint32_t vertexCount{ 0 };
    const interfaces::Material* material{ nullptr };
    interfaces::BoundingBox bb;

    Primitive() = default;
    Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, const interfaces::Material* material, interfaces::BoundingBox bb)
        : firstIndex(firstIndex), indexCount(indexCount), vertexCount(vertexCount), material(material), bb(bb)
    {}
};

}

#endif