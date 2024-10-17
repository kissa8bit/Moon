#ifndef GLTFMODEL_ANIMATION_H
#define GLTFMODEL_ANIMATION_H

#include <vector>
#include <limits>

#include "model.h"
#include "node.h"

#include "linearAlgebra.h"

namespace moon::models {

struct GltfAnimation : interfaces::Animation {
    struct Channel {
        enum PathType { TRANSLATION, ROTATION, SCALE } path;
        int samplerIndex{ tinygltf::invalidIndex };
        Node* node{nullptr};
    };
    using Channels = std::vector<Channel>;

    struct Point {
        using OutputData = std::vector<math::vec4>;
        float inputTime{ 0.0f };
        OutputData outputData;
    };
    using Points = std::vector<Point>;

    struct Sampler {
        enum InterpolationType { LINEAR, STEP, CUBICSPLINE } interpolation;
        Points points;
    };
    using Samplers = std::vector<Sampler>;

    RootNodes rootNodes;
    Channels channels;
    Samplers samplers;
    float totalTime{0};
    float changeTime{0};

    void setChangeTime(float changeTime) override;
    bool update(float time) override;
    float duration() const override;

    GltfAnimation(const RootNodes& rootNodes, const Channels& channels, const Samplers& samplers, float duration);
};

using GltfAnimations = std::vector<GltfAnimation>;

}

#endif