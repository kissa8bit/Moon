#ifndef GLTFMODEL_SKIN_H
#define GLTFMODEL_SKIN_H

#include <vector>

#include "matrix.h"

namespace moon::models {

struct Node;

struct Joint {
    math::Matrix<float, 4, 4> inverseBindMatrices{1.0f};
    Node* jointedNode{nullptr};
};

using Skin = std::vector<Joint>;
using Skins = std::vector<Skin>;

}

#endif