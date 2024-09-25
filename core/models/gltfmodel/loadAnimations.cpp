#include "gltfmodel.h"
#include "gltfutils.h"
#include "node.h"

#include <unordered_set>

namespace moon::models {

namespace {

template<typename Type>
GltfAnimation::Sampler makeSampler(
    GltfAnimation::Sampler::InterpolationType interpolation,
    size_t count,
    const float* inputData,
    const Type* outputData,
    float& duration)
{
    GltfAnimation::Sampler sampler{};
    sampler.interpolation = interpolation;
    for (size_t i = 0; i < count; ++i) {
        const auto& point = sampler.points.emplace_back(GltfAnimation::Point{inputData[i], outputData[i]});
        duration = std::max(point.inputTime, duration);
    }
    return sampler;
}

void translate(Node* node, const math::Vector<float,4>& x0, const math::Vector<float, 4>& x1, float t) {
    node->translation = mix(x0, x1, t).dvec();
}

void scale(Node* node, const math::Vector<float, 4>& x0, const math::Vector<float, 4>& x1, float t) {
    node->scale = mix(x0, x1, t).dvec();
}

void rotate(Node* node, const math::Vector<float, 4>& x0, const math::Vector<float, 4>& x1, float t) {
    math::Quaternion<float> q1(x0[3], x0.dvec());
    math::Quaternion<float> q2(x1[3], x1.dvec());
    node->rotation = normalize(slerp(q1, q2, t));
}

static const std::unordered_map<GltfAnimation::Channel::PathType, void (*)(Node*, const math::Vector<float, 4>&, const math::Vector<float, 4>&, float)> updateFMap = {
    {GltfAnimation::Channel::PathType::ROTATION, rotate},
    {GltfAnimation::Channel::PathType::TRANSLATION, translate},
    {GltfAnimation::Channel::PathType::SCALE, scale}
};

void translate(Node* node, const math::Vector<float, 4>& x, float t) {
    node->translation = mix(node->translation, math::Vector<float, 3>(x.dvec()), t);
}

void scale(Node* node, const math::Vector<float, 4>& x, float t) {
    node->scale = mix(node->scale, math::Vector<float, 3>(x.dvec()), t);
}

void rotate(Node* node, const math::Vector<float, 4>& x, float t) {
    math::Quaternion<float> q(x[3], x.dvec());
    node->rotation = normalize(slerp(node->rotation, q, t));
}

static const std::unordered_map<GltfAnimation::Channel::PathType, void (*)(Node*, const math::Vector<float, 4>&, float)> changeFMap = {
    {GltfAnimation::Channel::PathType::ROTATION, rotate},
    {GltfAnimation::Channel::PathType::TRANSLATION, translate},
    {GltfAnimation::Channel::PathType::SCALE, scale}
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

            if (!CHECK_M(input.count == output.count, "[ GltfModel::loadAnimations ] : input and output data must have same count")) continue;

            #define GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE, VEC_DIM)                                                                                           \
                case TINYGLTF_TYPE:                                                                                                                                         \
                    samplers.push_back(                                                                                                                                     \
                        makeSampler(interpolationMap.at(gltfsampler.interpolation), input.count, input.data, (const math::Vector<float, VEC_DIM>*)output.data, duration)    \
                    );                                                                                                                                                      \
                    break;

            switch (output.type) {
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC3, 3)
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC4, 4)
            }
            #undef GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE
        }

        for (auto& instance : instances) {
            GltfAnimation::ChannelsMap channelsMap;
            for (const auto &source: anim.channels) {
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    channelsMap[it->first].push_back(GltfAnimation::Channel{ channelsPathMap.at(source.target_path), source.sampler });
                }
            }
            instance.animations.push_back(GltfAnimation(&instance.nodes, channelsMap, samplers, duration));
        }
    }
}

GltfAnimation::GltfAnimation(NodeMap* nodes, const GltfAnimation::ChannelsMap& channelsMap, const GltfAnimation::Samplers& samplers, float duration)
    : nodes(nodes), channelsMap(channelsMap), samplers(samplers), dur(duration)
{}

bool GltfAnimation::change(float time, float changetime) {
    if (time > changetime) {
        return false;
    }

    for (const auto& [nodeIndex, channels] : channelsMap) {
        for (const auto& channel : channels) {
            const auto& sampler = samplers[channel.samplerIndex];
            float t = time / changetime;
            changeFMap.at(channel.path)(nodes->at(nodeIndex).get(), sampler.points[0].outputData, t);
        }
    }
    for (const auto& [_, node] : *nodes) {
        node->update();
    }
    return true;
}

bool GltfAnimation::update(float time){
    for (const auto& [nodeIndex, channels] : channelsMap) {
        for (const auto& channel : channels) {
            const auto& sampler = samplers[channel.samplerIndex];
            for (size_t i = 0; i < sampler.points.size() - 1; i++) {
                if (const auto& x0 = sampler.points[i], &x1 = sampler.points[i + 1]; time >= x0.inputTime && time <= x1.inputTime) {
                    const auto& x0t = x0.inputTime, & x1t = x1.inputTime;
                    const auto& x0d = x0.outputData, & x1d = x1.outputData;
                    const float t = (time - x0t) / (x1t - x0t);
                    updateFMap.at(channel.path)(nodes->at(nodeIndex).get(), x0d, x1d, t);
                }
            }
        }
    }
    for (const auto& [_, node] : *nodes) {
        node->update();
    }
    return true;
}

}
