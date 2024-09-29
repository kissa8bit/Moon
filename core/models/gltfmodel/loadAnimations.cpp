#include "gltfmodel.h"
#include "gltfutils.h"
#include "node.h"

namespace moon::models {

namespace {

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
        auto& point = sampler.points.emplace_back(GltfAnimation::Point{inputData[i], {}});
        point.outputData.resize(outputSize);
        for (size_t j = 0; j < outputSize; ++j) {
            point.outputData[j] = outputData[i * outputSize + j];
        }
        duration = std::max(point.inputTime, duration);
    }
    return sampler;
}

class Linear {
    using OutputData = GltfAnimation::Point::OutputData;

    static void translate(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->translation = mix(x0.at(0), x1.at(0), t).dvec();
    }

    static void scale(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->scale = mix(x0.at(0), x1.at(0), t).dvec();
    }

    static void rotate(Node* node, const OutputData& x0, const OutputData& x1, float t) {
        node->rotation = slerp(normalize(math::Quaternion<float>(x0.at(0)[3], x0.at(0).dvec())), normalize(math::Quaternion<float>(x1.at(0)[3], x1.at(0).dvec())), t);
    }

public:
    using signature = void (*)(Node*, const OutputData&, const OutputData&, float);

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
        node->translation = x0.at(0).dvec();
    }

    static void scale(Node* node, const OutputData& x0) {
        node->scale = x0.at(0).dvec();
    }

    static void rotate(Node* node, const OutputData& x0) {
        node->rotation = math::Quaternion<float>(x0.at(0)[3], x0.at(0).dvec());
    }

public:
    using signature = void (*)(Node*, const OutputData&);

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

    static void translate(Node* node, const OutputData& x, float t) {
        node->translation = mix(node->translation, math::Vector<float, 3>(x.at(0).dvec()), t);
    }

    static void scale(Node* node, const OutputData& x, float t) {
        node->scale = mix(node->scale, math::Vector<float, 3>(x.at(0).dvec()), t);
    }

    static void rotate(Node* node, const OutputData& x, float t) {
        math::Quaternion<float> q(x.at(0)[3], x.at(0).dvec());
        node->rotation = normalize(slerp(node->rotation, q, t));
    }

public:
    using signature = void (*)(Node*, const OutputData&, float);

    static signature update(GltfAnimation::Channel::PathType path) {
        static const std::unordered_map<GltfAnimation::Channel::PathType, signature> update = {
            {GltfAnimation::Channel::PathType::ROTATION, Change::rotate},
            {GltfAnimation::Channel::PathType::TRANSLATION, Change::translate},
            {GltfAnimation::Channel::PathType::SCALE, Change::scale}
        };
        return update.at(path);
    }
};

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
            #undef GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE
        }

        for (auto& instance : instances) {
            GltfAnimation::Channels channels;
            for (const auto &source: anim.channels) {
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    const auto& [_, node] = *it;
                    channels.push_back(GltfAnimation::Channel{ channelsPathMap.at(source.target_path), source.sampler, node.get() });
                }
            }
            instance.animations.push_back(GltfAnimation(instance.rootNodes, channels, samplers, duration));
        }
    }
}

GltfAnimation::GltfAnimation(const RootNodes& rootNodes, const GltfAnimation::Channels& channels, const GltfAnimation::Samplers& samplers, float duration)
    : rootNodes(rootNodes), channels(channels), samplers(samplers), totalTime(duration)
{}

void GltfAnimation::setChangeTime(float time) {
    changeTime = time;
}

bool GltfAnimation::update(float time){
    bool needUpdate = false;
    if (time < changeTime) {
        for (const auto& channel : channels) {
            if (channel.samplerIndex >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            float t = time / changeTime;
            Change::update(channel.path)(channel.node, sampler.points[0].outputData, t);
        }
        needUpdate |= true;
    } else {
        for (const auto& channel : channels) {
            if(channel.samplerIndex >= samplers.size()) continue;
            const auto& sampler = samplers[channel.samplerIndex];
            for (size_t i = 0; i < sampler.points.size() - 1; i++) {
                if (const auto& x0 = sampler.points[i], &x1 = sampler.points[i + 1]; time > x0.inputTime && time < x1.inputTime) {
                    const auto& x0t = x0.inputTime, & x1t = x1.inputTime;
                    const auto& x0d = x0.outputData, & x1d = x1.outputData;
                    const float t = (time - x0t) / (x1t - x0t);
                    switch (sampler.interpolation) {
                        case Sampler::InterpolationType::STEP: Step::update(channel.path)(channel.node, x0d); break;
                        case Sampler::InterpolationType::LINEAR: Linear::update(channel.path)(channel.node, x0d, x1d, t); break;
                        case Sampler::InterpolationType::CUBICSPLINE: Linear::update(channel.path)(channel.node, x0d, x1d, t); break;
                    }
                    needUpdate |= true;
                }
            }
        }
    }
    if (needUpdate) updateRootNodes(rootNodes);
    return needUpdate;
}

float GltfAnimation::duration() const {
    return totalTime;
}

}
