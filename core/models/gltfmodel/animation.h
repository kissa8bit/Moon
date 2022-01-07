#ifndef MOON_MODELS_GLTFMODEL_ANIMATION_H
#define MOON_MODELS_GLTFMODEL_ANIMATION_H

#include <vector>

#include <math/linearAlgebra.h>

#include "node.h"
#include "gltfskeleton.h"

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

    Nodes* nodeMap{ nullptr };
    GltfSkeletons* skeletons{ nullptr };
    Channels channels;
    Samplers samplers;
    float totalTime{0};
    float changeTime{0};

    void setChangeTime(float changeTime) override;
    bool update(float time) override;
    float duration() const override;

    GltfAnimation(Nodes* nodeMap, GltfSkeletons* skeletons, const Channels& channels, const Samplers& samplers, float duration);
};

using GltfAnimations = std::vector<GltfAnimation>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_ANIMATION_H