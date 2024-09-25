#ifndef GLTFMODEL_ANIMATION_H
#define GLTFMODEL_ANIMATION_H

#include <vector>
#include <limits>

#include "vector.h"
#include "model.h"
#include "node.h"

namespace moon::models {

struct GltfAnimation : interfaces::Animation {
    struct Channel {
        enum PathType { TRANSLATION, ROTATION, SCALE } path;
        int samplerIndex{ tinygltf::invalidIndex };
    };
    using ChannelsMap = std::unordered_map<uint32_t, std::vector<Channel>>;

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

    NodeMap* nodes{nullptr};
    ChannelsMap channelsMap;
    Samplers samplers;
    float dur{0};

    bool change(float time, float changetime) override;
    bool update(float time) override;
    float duration() const override {return dur;}

    GltfAnimation(NodeMap* nodes, const ChannelsMap& channelsMap, const Samplers& samplers, float duration);
};

using GltfAnimations = std::vector<GltfAnimation>;

}

#endif