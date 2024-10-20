#ifndef GLTFMODEL_SKIN_H
#define GLTFMODEL_SKIN_H

#include <vector>

#include "linearAlgebra.h"

namespace moon::models {

struct Joint {
    math::mat4 inverseBindMatrices{ math::mat4::identity() };
    NodeId jointedNodeId{ invalidNodeId };
};

using Skin = std::vector<Joint>;
using Skins = std::vector<Skin>;

}

#endif