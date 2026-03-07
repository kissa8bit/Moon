#ifndef MOON_MODELS_GLTFMODEL_ANIMATION_H
#define MOON_MODELS_GLTFMODEL_ANIMATION_H

#include <vector>
#include <unordered_set>

#include <math/linearAlgebra.h>

#include "node.h"
#include "gltfskeleton.h"
#include "gltfmorph.h"

namespace moon::models {

struct GltfAnimation : interfaces::Animation {
    struct Channel {
        enum PathType { TRANSLATION, ROTATION, SCALE, WEIGHTS } path;
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

    struct PoseSnapshot {
        math::vec3 translation;
        math::vec3 scale;
        math::quat rotation;
        std::vector<float> weights;
    };

    Nodes* nodeMap{ nullptr };
    GltfSkeletons* skeletons{ nullptr };
    GltfMorphWeightsMap* morphWeights{ nullptr };
    Channels channels;
    Samplers samplers;
    float totalTime{0};
    float changeTime{0};
    std::unordered_set<Node*> animatedNodes;
    std::unordered_map<Node*, PoseSnapshot> blendStartPose;
    bool initialized{false};
    bool blendStartCaptured{false};

    void setChangeTime(float changeTime) override;
    bool applyChannels(float time) override;
    void updateNodes() override;
    bool update(float time) override;
    float duration() const override;

    GltfAnimation(Nodes* nodeMap, GltfSkeletons* skeletons, GltfMorphWeightsMap* morphWeights, const Channels& channels, const Samplers& samplers, float duration);
};

using GltfAnimations = std::vector<GltfAnimation>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_ANIMATION_H