#ifndef GLTFMODEL_ANIMATION_H
#define GLTFMODEL_ANIMATION_H

#include <vector>
#include <limits>

#include "vector.h"

namespace moon::models {

struct Node;

struct Animation {
    struct AnimationChannel {
        enum PathType { TRANSLATION, ROTATION, SCALE };
        PathType path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler {
        enum InterpolationType { LINEAR, STEP, CUBICSPLINE };
        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<math::Vector<float, 4>> outputsVec4;
    };

    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();
};

using Animations = std::vector<Animation>;

}

#endif