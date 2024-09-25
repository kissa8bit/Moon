#ifndef GLTFMODEL_INSTANCE_H
#define GLTFMODEL_INSTANCE_H

#include <vector>

#include "node.h"
#include "skin.h"
#include "animation.h"

namespace moon::models {

struct Instance {
    NodeMap nodes;
    Skins skins;
    GltfAnimations animations;

    Instance() = default;
    Instance(Instance&& other) noexcept {
        swap(other);
    }
    Instance& operator=(Instance&& other) noexcept {
        swap(other);
        return *this;
    }
    void swap(Instance& other) noexcept {
        std::swap(nodes, other.nodes);
        std::swap(skins, other.skins);
        std::swap(animations, other.animations);
    }
};

using Instances = std::vector<Instance>;

}

#endif