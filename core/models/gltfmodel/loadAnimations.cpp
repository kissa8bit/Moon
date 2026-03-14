#include "gltfmodel.h"
#include "gltfutils.h"
#include "node.h"

#include <math/linearAlgebra.h>

namespace moon::models {

namespace {

template<typename type>
math::Quaternion<type> quatFromVec(const math::Vector<type, 4>& vec) {
    return normalize(math::Quaternion<type>(vec[3], vec.dvec()));
}

class Linear {
    using OutputData = GltfAnimation::Point::OutputData;

    static void translate(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->translation = mix(x0.at(value), x1.at(value), t).dvec();
    }

    static void scale(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->scale = mix(x0.at(value), x1.at(value), t).dvec();
    }

    static void rotate(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->rotation = slerp(quatFromVec(x0.at(value)), quatFromVec(x1.at(value)), t);
    }

public:
    using signature = void (*)(Node*, const OutputData&, const OutputData&, float);
    static constexpr size_t value = 0;

    static signature update(GltfAnimation::Channel::PathType path) {
        static const std::unordered_map<GltfAnimation::Channel::PathType, signature> update = {
            {GltfAnimation::Channel::PathType::ROTATION, rotate},
            {GltfAnimation::Channel::PathType::TRANSLATION, translate},
            {GltfAnimation::Channel::PathType::SCALE, scale}
        };
        return update.at(path);
    }
};

class Step {
    using OutputData = GltfAnimation::Point::OutputData;

    static void translate(Node* node, const OutputData& x0) {
        node->translation = x0.at(value).dvec();
    }

    static void scale(Node* node, const OutputData& x0) {
        node->scale = x0.at(value).dvec();
    }

    static void rotate(Node* node, const OutputData& x0) {
        node->rotation = quatFromVec(x0.at(value));
    }

public:
    using signature = void (*)(Node*, const OutputData&);
    static constexpr size_t value = 0;

    static signature update(GltfAnimation::Channel::PathType path) {
        static const std::unordered_map<GltfAnimation::Channel::PathType, signature> update = {
            {GltfAnimation::Channel::PathType::ROTATION, rotate},
            {GltfAnimation::Channel::PathType::TRANSLATION, translate},
            {GltfAnimation::Channel::PathType::SCALE, scale}
        };
        return update.at(path);
    }
};

class Cubic {
    using OutputData = GltfAnimation::Point::OutputData;

    static math::vec4 splineInterpolation(const OutputData& x0, const OutputData& x1, float t0, float t1, float t) {
        const float t2 = t * t, t3 = t2 * t, delta = t1 - t0;
        math::vec4 result{ 0.0f };
        for (uint32_t i = 0; i < 4; i++) {
            const float p0 = x0.at(value)[i], d0 = delta * x0.at(tangent.a)[i];
            const float p1 = x1.at(value)[i], d1 = delta* x1.at(tangent.b)[i];
            result[i] =
                (2.0f * t3 - 3.0f * t2 + 1.0f) * p0 +
                (-2.0f * t3 + 3.0f * t2) * p1 +
                (t3 - 2.0f * t2 + t) * d0 +
                (t3 - t2) * d1;
        }
        return result;
    }

    static void translate(Node* node, const OutputData& x0, const OutputData& x1, float t0, float t1, float t) {
        const auto v = splineInterpolation(x0, x1, t0, t1, t);
        node->translation = v.dvec();
    }

    static void scale(Node* node, const OutputData& x0, const OutputData& x1, float t0, float t1, float t) {
        const auto v = splineInterpolation(x0, x1, t0, t1, t);
        node->scale = v.dvec();
    }

    static void rotate(Node* node, const OutputData& x0, const OutputData& x1, float t0, float t1, float t) {
        const auto v = splineInterpolation(x0, x1, t0, t1, t);
        node->rotation = quatFromVec(v);
    }

public:
    using signature = void (*)(Node*, const OutputData&, const OutputData&, float, float, float);
    static constexpr struct { size_t a; size_t b; } tangent = { 2, 0 };  // a=outTangent(p0), b=inTangent(p1)
    static constexpr size_t value = 1;

    static signature update(GltfAnimation::Channel::PathType path) {
        static const std::unordered_map<GltfAnimation::Channel::PathType, signature> update = {
            {GltfAnimation::Channel::PathType::ROTATION, rotate},
            {GltfAnimation::Channel::PathType::TRANSLATION, translate},
            {GltfAnimation::Channel::PathType::SCALE, scale}
        };
        return update.at(path);
    }
};

static size_t getKeyframeValueIndex(GltfAnimation::Sampler::InterpolationType interpolation) {
    return interpolation == GltfAnimation::Sampler::InterpolationType::CUBICSPLINE ? Cubic::value : Linear::value;
}

static void applyWeights(Node* node, const GltfAnimation::Point::OutputData& data) {
    node->weights.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        node->weights[i] = data[i][0];
    }
}

static void lerpWeights(Node* node, const GltfAnimation::Point::OutputData& x0, const GltfAnimation::Point::OutputData& x1, float t) {
    const size_t count = std::min(x0.size(), x1.size());
    node->weights.resize(count);
    for (size_t i = 0; i < count; i++) {
        node->weights[i] = x0[i][0] + t * (x1[i][0] - x0[i][0]);
    }
}

template<typename Type>
GltfAnimation::Sampler makeSampler(
    GltfAnimation::Sampler::InterpolationType interpolation,
    size_t count,
    size_t outputSize,
    const float* inputData,
    const Type* outputData,
    float& duration)
{
    GltfAnimation::Sampler sampler{};
    sampler.interpolation = interpolation;
    for (size_t i = 0; i < count; ++i) {
        auto& point = sampler.points.emplace_back(GltfAnimation::Point{ inputData[i], {} });
        point.outputData.resize(outputSize);
        for (size_t j = 0; j < outputSize; ++j) {
            point.outputData[j] = outputData[i * outputSize + j];
        }
        duration = std::max(point.inputTime, duration);
    }
    return sampler;
}

}

void GltfModel::loadAnimations(const tinygltf::Model& gltfModel){
    static const std::unordered_map<std::string, GltfAnimation::Sampler::InterpolationType> interpolationMap = {
        {"LINEAR", GltfAnimation::Sampler::InterpolationType::LINEAR},
        {"STEP", GltfAnimation::Sampler::InterpolationType::STEP},
        {"CUBICSPLINE", GltfAnimation::Sampler::InterpolationType::CUBICSPLINE}
    };
    static const std::unordered_map<std::string, GltfAnimation::Channel::PathType> channelsPathMap = {
        {"rotation", GltfAnimation::Channel::PathType::ROTATION},
        {"translation", GltfAnimation::Channel::PathType::TRANSLATION},
        {"scale", GltfAnimation::Channel::PathType::SCALE},
        {"weights", GltfAnimation::Channel::PathType::WEIGHTS}
    };

    for (const tinygltf::Animation &anim : gltfModel.animations) {
        m_animationNames.push_back(anim.name);
        GltfAnimation::Samplers samplers;
        float duration{ 0 };

        for (const auto& gltfsampler : anim.samplers) {
            const GltfBufferExtractor<const float*> input(gltfModel, gltfsampler.input);
            const GltfBufferExtractor output(gltfModel, gltfsampler.output);

            if (!CHECK_M(input.count <= output.count, "[ GltfModel::loadAnimations ] : the input data less than the output data")) continue;
            if (!CHECK_M(output.count % input.count == 0, "[ GltfModel::loadAnimations ] : the output data must be a multiple of the input data without remainder")) continue;

            const size_t outputSize = output.count / input.count;
            #define GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE, VEC_DIM)   \
                case TINYGLTF_TYPE:                                                 \
                    samplers.push_back(                                             \
                        makeSampler(                                                \
                            interpolationMap.at(gltfsampler.interpolation),         \
                            input.count,                                            \
                            outputSize,                                             \
                            input.data,                                             \
                            (const math::Vector<float, VEC_DIM>*)output.data,       \
                            duration)                                               \
                    );                                                                                                                                                                  \
                    break;

            switch (output.type) {
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC3, 3)
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC4, 4)
                case TINYGLTF_TYPE_SCALAR: {
                    // Weight animation: outputSize scalars per keyframe, each stored as vec4.x
                    const float* scalarData = (const float*)output.data;
                    GltfAnimation::Sampler sampler{};
                    sampler.interpolation = interpolationMap.at(gltfsampler.interpolation);
                    for (size_t i = 0; i < input.count; ++i) {
                        auto& point = sampler.points.emplace_back(GltfAnimation::Point{input.data[i], {}});
                        point.outputData.resize(outputSize);
                        for (size_t j = 0; j < outputSize; ++j) {
                            point.outputData[j] = math::vec4(scalarData[i * outputSize + j], 0.0f, 0.0f, 0.0f);
                        }
                        duration = std::max(point.inputTime, duration);
                    }
                    samplers.push_back(std::move(sampler));
                    break;
                }
            }
        }

        for (auto& instance : instances) {
            GltfAnimation::Channels channels;
            for (const auto &source: anim.channels) {
                const auto pathIt = channelsPathMap.find(source.target_path);
                if (pathIt == channelsPathMap.end()) continue;
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    auto& [_, node] = *it;
                    channels.push_back(GltfAnimation::Channel{ pathIt->second, source.sampler, &node });
                }
            }
            instance.animations.push_back(GltfAnimation(&instance.nodes, &instance.meshNodes, channels, samplers, duration));
        }
    }
}

GltfAnimation::GltfAnimation(Nodes* nodeMap, GltfNodeDataMap* meshNodes, const GltfAnimation::Channels& channels, const GltfAnimation::Samplers& samplers, float duration)
    : nodeMap(nodeMap), meshNodes(meshNodes), channels(channels), samplers(samplers), totalTime(duration)
{
    for (const auto& channel : this->channels) {
        if (channel.node) animatedNodes.insert(channel.node);
    }
}

void GltfAnimation::setChangeTime(float time) {
    changeTime = time;
    initialized = false;
    blendStartCaptured = false;
}

bool GltfAnimation::applyChannels(float time){
    bool needUpdate = false;
    if (time < changeTime) {
        // Capture all node poses once at the start of each blend (initial switch or loop seam)
        if (!blendStartCaptured) {
            blendStartCaptured = true;
            blendStartPose.clear();
            for (auto& [_, node] : *nodeMap) {
                blendStartPose[&node] = { node.translation, node.scale, node.rotation, node.weights };
            }
        }
        const float t = time / changeTime;
        // Blend channel nodes: from captured start pose → keyframe[0] of new animation
        for (const auto& channel : channels) {
            if (channel.samplerIndex < 0 || static_cast<size_t>(channel.samplerIndex) >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            if (sampler.points.empty()) continue;
            const auto snapIt = blendStartPose.find(channel.node);
            if (snapIt == blendStartPose.end()) continue;
            const auto& snap = snapIt->second;
            const auto& target = sampler.points[0].outputData.at(getKeyframeValueIndex(sampler.interpolation));
            switch (channel.path) {
                case Channel::PathType::TRANSLATION:
                    channel.node->translation = mix(snap.translation, math::vec3(target.dvec()), t); break;
                case Channel::PathType::SCALE:
                    channel.node->scale = mix(snap.scale, math::vec3(target.dvec()), t); break;
                case Channel::PathType::ROTATION:
                    channel.node->rotation = normalize(slerp(snap.rotation, quatFromVec(target), t)); break;
                case Channel::PathType::WEIGHTS: {
                    const auto& targetData = sampler.points[0].outputData;
                    const size_t count = targetData.size();
                    channel.node->weights.resize(count);
                    for (size_t i = 0; i < count; i++) {
                        const float startW = i < snap.weights.size() ? snap.weights[i] : 0.0f;
                        channel.node->weights[i] = startW + t * (targetData[i][0] - startW);
                    }
                    break;
                }
            }
        }
        // Blend non-channel nodes: from captured start pose → rest pose
        for (auto& [_, node] : *nodeMap) {
            if (animatedNodes.count(&node) == 0) {
                const auto& snap = blendStartPose.at(&node);
                node.translation = mix(snap.translation, node.restTranslation, t);
                node.scale = mix(snap.scale, node.restScale, t);
                node.rotation = normalize(slerp(snap.rotation, node.restRotation, t));
            }
        }
        needUpdate = true;
    } else {
        blendStartCaptured = false; // reset so next loop captures fresh start pose
        // On first frame after the blend, snap non-animated nodes to rest pose
        if (!initialized) {
            initialized = true;
            changeTime = 0; // blend is one-shot: applies only on the initial switch, not on every loop
            for (auto& [_, node] : *nodeMap) {
                if (animatedNodes.count(&node) == 0) {
                    node.translation = node.restTranslation;
                    node.scale = node.restScale;
                    node.rotation = node.restRotation;
                    needUpdate = true;
                }
            }
        }
        for (const auto& channel : channels) {
            if(channel.samplerIndex < 0 || static_cast<size_t>(channel.samplerIndex) >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            if (sampler.points.size() < 2) {
                if (!sampler.points.empty()) {
                    if (channel.path == Channel::PathType::WEIGHTS) {
                        applyWeights(channel.node, sampler.points[0].outputData);
                    } else {
                        Step::update(channel.path)(channel.node, sampler.points[0].outputData);
                    }
                    needUpdate = true;
                }
                continue;
            }
            bool applied = false;
            for (size_t i = 0; i < sampler.points.size() - 1; i++) {
                if (const auto& x0 = sampler.points[i], &x1 = sampler.points[i + 1]; time >= x0.inputTime && time < x1.inputTime) {
                    const auto x0t = x0.inputTime, x1t = x1.inputTime;
                    const auto& x0d = x0.outputData, &x1d = x1.outputData;
                    const float t = (time - x0t) / (x1t - x0t);
                    if (channel.path == Channel::PathType::WEIGHTS) {
                        switch (sampler.interpolation) {
                            case Sampler::InterpolationType::STEP: applyWeights(channel.node, x0d); break;
                            case Sampler::InterpolationType::LINEAR:
                            case Sampler::InterpolationType::CUBICSPLINE:
                                lerpWeights(channel.node, x0d, x1d, t); break;
                        }
                    } else {
                        switch (sampler.interpolation) {
                            case Sampler::InterpolationType::STEP: Step::update(channel.path)(channel.node, x0d); break;
                            case Sampler::InterpolationType::LINEAR: Linear::update(channel.path)(channel.node, x0d, x1d, t); break;
                            case Sampler::InterpolationType::CUBICSPLINE: Cubic::update(channel.path)(channel.node, x0d, x1d, x0t, x1t, t); break;
                        }
                    }
                    applied = true;
                    needUpdate = true;
                    break;
                }
            }
            // Hold last frame when time is past the final keyframe
            if (!applied && time >= sampler.points.back().inputTime) {
                if (channel.path == Channel::PathType::WEIGHTS) {
                    applyWeights(channel.node, sampler.points.back().outputData);
                } else {
                    Step::update(channel.path)(channel.node, sampler.points.back().outputData);
                }
                needUpdate = true;
            }
        }
    }
    return needUpdate;
}

void GltfAnimation::updateNodes() {
    moon::models::updateNodes(*nodeMap, *meshNodes);
}

bool GltfAnimation::update(float time){
    const bool needUpdate = applyChannels(time);
    if (needUpdate) {
        updateNodes();
    }
    return needUpdate;
}

float GltfAnimation::duration() const {
    return totalTime;
}

} // moon::models
