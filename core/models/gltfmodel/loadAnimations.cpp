#include "gltfmodel.h"
#include "gltfutils.h"
#include "node.h"

namespace moon::models {

namespace {

template<typename Type>
Animation::Sampler makeSampler(
    Animation::Sampler::InterpolationType interpolation,
    size_t count,
    const float* inputData,
    const Type* outputData,
    float& start,
    float& end)
{
    Animation::Sampler sampler{};
    sampler.interpolation = interpolation;
    for (size_t i = 0; i < count; ++i) {
        const auto& point = sampler.points.emplace_back(Animation::Point{inputData[i], outputData[i]});
        start = std::min(point.inputTime, start);
        end = std::max(point.inputTime, end);
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

static const std::unordered_map<Animation::Channel::PathType, void (*)(Node*, const math::Vector<float, 4>&, const math::Vector<float, 4>&, float)> updateFMap = {
    {Animation::Channel::PathType::ROTATION, rotate},
    {Animation::Channel::PathType::TRANSLATION, translate},
    {Animation::Channel::PathType::SCALE, scale}
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

static const std::unordered_map<Animation::Channel::PathType, void (*)(Node*, const math::Vector<float, 4>&, float)> changeFMap = {
    {Animation::Channel::PathType::ROTATION, rotate},
    {Animation::Channel::PathType::TRANSLATION, translate},
    {Animation::Channel::PathType::SCALE, scale}
};

}

bool GltfModel::hasAnimation(uint32_t frameIndex) const {
    return instances[instances.size() > frameIndex ? frameIndex : 0].animations.size() > 0;
}

float GltfModel::animationStart(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].start;
}

float GltfModel::animationEnd(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].end;
}

void GltfModel::loadAnimations(const tinygltf::Model& gltfModel){
    static const std::unordered_map<std::string, Animation::Sampler::InterpolationType> interpolationMap = {
        {"LINEAR", Animation::Sampler::InterpolationType::LINEAR},
        {"STEP", Animation::Sampler::InterpolationType::STEP},
        {"CUBICSPLINE", Animation::Sampler::InterpolationType::CUBICSPLINE}
    };
    static const std::unordered_map<std::string, Animation::Channel::PathType> channelsMap = {
        {"rotation", Animation::Channel::PathType::ROTATION},
        {"translation", Animation::Channel::PathType::TRANSLATION},
        {"scale", Animation::Channel::PathType::SCALE}
    };

    for (const tinygltf::Animation &anim : gltfModel.animations) {
        Animation::Samplers samplers;
        float start{ std::numeric_limits<float>::max() };
        float end{ std::numeric_limits<float>::min() };

        for (const auto& gltfsampler : anim.samplers) {
            const GltfBufferExtractor<const float*> input(gltfModel, gltfsampler.input);
            const GltfBufferExtractor output(gltfModel, gltfsampler.output);

            if (!CHECK_M(input.count == output.count, "[ GltfModel::loadAnimations ] : input and output data must have same count")) continue;

            #define GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE, VEC_DIM)                                                                                           \
                case TINYGLTF_TYPE:                                                                                                                                         \
                    samplers.push_back(                                                                                                                                     \
                        makeSampler(interpolationMap.at(gltfsampler.interpolation), input.count, input.data, (const math::Vector<float, VEC_DIM>*)output.data, start, end)  \
                    );                                                                                                                                                      \
                    break;

            switch (output.type) {
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC3, 3)
                GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE(TINYGLTF_TYPE_VEC4, 4)
            }
            #undef GLTFMODEL_LOADANIMATIONS_SAMPLER_CASE
        }

        for (auto& instance : instances) {
            Animation::Channels channels;
            for (const auto &source: anim.channels) {
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    channels.push_back(Animation::Channel{ channelsMap.at(source.target_path) , it->second.get(), source.sampler });
                }
            }
            instance.animations.push_back({ channels, samplers, start, end });
        }
    }
}

void GltfModel::updateAnimation(uint32_t instanceIndex, uint32_t index, float time)
{
    bool update = false;
    const auto& animation = instances.at(instanceIndex).animations.at(index);
    for (const auto& channel : animation.channels) {
        const Animation::Sampler& sampler = animation.samplers[channel.samplerIndex];

        auto left = sampler.points.begin(), right = std::next(left);
        for (; right != sampler.points.end(); right = std::next(right), left = std::next(left)) {
            if (time >= left->inputTime && time <= right->inputTime) break;
        }

        if(right == sampler.points.end()) continue;
        update |= true;

        const auto& x0 = *left, & x1 = *right;
        const auto& x0t = x0.inputTime, & x1t = x1.inputTime;
        const auto& x0d = x0.outputData, & x1d = x1.outputData;

        const float t = (time - x0t) / (x1t - x0t);
        updateFMap.at(channel.path)(channel.node, x0d, x1d, t);
    }
    if (update) {
        for (auto& [_, node] : instances.at(instanceIndex).nodes) {
            node->update();
        }
    }
}

void GltfModel::changeAnimation(uint32_t instanceIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime)
{
    const auto& animation = instances.at(instanceIndex).animations.at(newIndex);
    for (const auto& channel : animation.channels) {
        const auto& sampler = animation.samplers[channel.samplerIndex];
        float t = (time - startTime) / changeAnimationTime;
        changeFMap.at(channel.path)(channel.node, sampler.points[0].outputData, t);
    }
    for (auto& [_, node] : instances.at(instanceIndex).nodes) {
        node->update();
    }
}

}
