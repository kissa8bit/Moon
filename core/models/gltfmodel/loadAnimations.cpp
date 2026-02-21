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

class Change {
    using OutputData = GltfAnimation::Point::OutputData;

    static size_t getValue(GltfAnimation::Sampler::InterpolationType interpolation) {
        static const std::unordered_map<GltfAnimation::Sampler::InterpolationType, size_t> value = {
            {GltfAnimation::Sampler::InterpolationType::CUBICSPLINE, Cubic::value},
            {GltfAnimation::Sampler::InterpolationType::LINEAR, Linear::value},
            {GltfAnimation::Sampler::InterpolationType::STEP, Step::value}
        };
        return value.at(interpolation);
    }

    static void translate(Node* node, GltfAnimation::Sampler::InterpolationType interpolation, const OutputData& x, float t) {
        const auto value = getValue(interpolation);
        node->translation = mix(node->translation, math::vec3(x.at(value).dvec()), t);
    }

    static void scale(Node* node, GltfAnimation::Sampler::InterpolationType interpolation, const OutputData& x, float t) {
        const auto value = getValue(interpolation);
        node->scale = mix(node->scale, math::vec3(x.at(value).dvec()), t);
    }

    static void rotate(Node* node, GltfAnimation::Sampler::InterpolationType interpolation, const OutputData& x, float t) {
        const auto value = getValue(interpolation);
        node->rotation = normalize(slerp(node->rotation, quatFromVec(x.at(value)), t));
    }

public:
    using signature = void (*)(Node*, GltfAnimation::Sampler::InterpolationType, const OutputData&, float);

    static signature update(GltfAnimation::Channel::PathType path) {
        static const std::unordered_map<GltfAnimation::Channel::PathType, signature> update = {
            {GltfAnimation::Channel::PathType::ROTATION, Change::rotate},
            {GltfAnimation::Channel::PathType::TRANSLATION, Change::translate},
            {GltfAnimation::Channel::PathType::SCALE, Change::scale}
        };
        return update.at(path);
    }
};

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
        {"scale", GltfAnimation::Channel::PathType::SCALE}
    };

    for (const tinygltf::Animation &anim : gltfModel.animations) {
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
            }
        }

        for (auto& instance : instances) {
            GltfAnimation::Channels channels;
            for (const auto &source: anim.channels) {
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    auto& [_, node] = *it;
                    channels.push_back(GltfAnimation::Channel{ channelsPathMap.at(source.target_path), source.sampler, &node });
                }
            }
            instance.animations.push_back(GltfAnimation(&instance.nodes, &instance.skeletons, channels, samplers, duration));
        }
    }
}

GltfAnimation::GltfAnimation(Nodes* nodeMap, GltfSkeletons* skeletons, const GltfAnimation::Channels& channels, const GltfAnimation::Samplers& samplers, float duration)
    : nodeMap(nodeMap), skeletons(skeletons), channels(channels), samplers(samplers), totalTime(duration)
{}

void GltfAnimation::setChangeTime(float time) {
    changeTime = time;
}

bool GltfAnimation::update(float time){
    bool needUpdate = false;
    if (time < changeTime) {
        for (const auto& channel : channels) {
            if (channel.samplerIndex < 0 || static_cast<size_t>(channel.samplerIndex) >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            if (sampler.points.size() == 0) continue;
            float t = time / changeTime;
            Change::update(channel.path)(channel.node, sampler.interpolation, sampler.points.at(0).outputData, t);
        }
        needUpdate |= true;
    } else {
        for (const auto& channel : channels) {
            if(channel.samplerIndex < 0 || static_cast<size_t>(channel.samplerIndex) >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            for (size_t i = 0; i < sampler.points.size() - 1; i++) {
                if (const auto& x0 = sampler.points[i], &x1 = sampler.points[i + 1]; time > x0.inputTime && time < x1.inputTime) {
                    const auto& x0t = x0.inputTime, & x1t = x1.inputTime;
                    const auto& x0d = x0.outputData, & x1d = x1.outputData;
                    const float t = (time - x0t) / (x1t - x0t);
                    switch (sampler.interpolation) {
                        case Sampler::InterpolationType::STEP: Step::update(channel.path)(channel.node, x0d); break;
                        case Sampler::InterpolationType::LINEAR: Linear::update(channel.path)(channel.node, x0d, x1d, t); break;
                        case Sampler::InterpolationType::CUBICSPLINE: Cubic::update(channel.path)(channel.node, x0d, x1d, x0t, x1t, t); break;
                    }
                    needUpdate = true;
                }
            }
        }
    }
    if (needUpdate){
        updateNodes(*nodeMap, *skeletons);
    }
    return needUpdate;
}

float GltfAnimation::duration() const {
    return totalTime;
}

} // moon::models
