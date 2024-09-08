#ifndef GLTFMODEL_SKIN_H
#define GLTFMODEL_SKIN_H

#include <vector>

#include "matrix.h"

namespace moon::models {

struct Node;

struct Skin {
    std::vector<math::Matrix<float, 4, 4>> inverseBindMatrices;
    std::vector<Node*> joints;
};

using Skins = std::vector<Skin>;

}

#endif