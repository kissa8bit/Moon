#ifndef GLTFMODEL_ANIMATION_H
#define GLTFMODEL_ANIMATION_H

#include <vector>
#include <limits>

#include "vector.h"

namespace moon::models {

struct Node;

struct Animation {
    struct Channel {
        enum PathType { TRANSLATION, ROTATION, SCALE } path;
        Node* node{ nullptr };
        int samplerIndex{ tinygltf::invalidIndex };
    };
    using Channels = std::vector<Channel>;

    struct Point {
        float inputTime{ 0.0f };
        math::Vector<float, 4> outputData{ 0.0f };
    };
    using Points = std::vector<Point>;

    struct Sampler {
        enum InterpolationType { LINEAR, STEP, CUBICSPLINE } interpolation;
        Points points;
    };
    using Samplers = std::vector<Sampler>;

    Channels channels;
    Samplers samplers;
    float start{ std::numeric_limits<float>::max() };
    float end{ std::numeric_limits<float>::min() };
};

using Animations = std::vector<Animation>;

}

#endif