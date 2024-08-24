#include "gltfmodel.h"

namespace moon::models {

bool GltfModel::hasAnimation(uint32_t frameIndex) const {
    return instances[instances.size() > frameIndex ? frameIndex : 0].animations.size() > 0;
}

float GltfModel::animationStart(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].start;
}

float GltfModel::animationEnd(uint32_t frameIndex, uint32_t index) const {
    return instances[frameIndex].animations[index].end;
}

void GltfModel::loadAnimations(const tinygltf::Model& gltfModel)
{
    for(auto& instance: instances){
        for (const tinygltf::Animation &anim : gltfModel.animations) {
            Animation animation{};

            // Samplers
            for (const auto &samp : anim.samplers) {
                Animation::AnimationSampler sampler{};

                if (samp.interpolation == "LINEAR") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::LINEAR;
                }
                if (samp.interpolation == "STEP") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::STEP;
                }
                if (samp.interpolation == "CUBICSPLINE") {
                    sampler.interpolation = Animation::AnimationSampler::InterpolationType::CUBICSPLINE;
                }

                // Read sampler input time values
                {
                    const tinygltf::Accessor &accessor = gltfModel.accessors[samp.input];
                    const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float *buf = static_cast<const float*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++) {
                        sampler.inputs.push_back(buf[index]);
                    }

                    for (const auto& input: sampler.inputs) {
                        animation.start = std::min(input, animation.start);
                        animation.end = std::max(input, animation.end);
                    }
                }

                // Read sampler output T/R/S values
                {
                    const tinygltf::Accessor &accessor = gltfModel.accessors[samp.output];
                    const tinygltf::BufferView &bufferView = gltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer &buffer = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    switch (accessor.type) {
                        case TINYGLTF_TYPE_VEC3: {
                            const moon::math::Vector<float,3> *buf = static_cast<const moon::math::Vector<float,3>*>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++) {
                                sampler.outputsVec4.push_back(moon::math::Vector<float,4>(buf[index][0],buf[index][1],buf[index][2], 0.0f));
                            }
                            break;
                        }
                        case TINYGLTF_TYPE_VEC4: {
                            const moon::math::Vector<float,4> *buf = static_cast<const moon::math::Vector<float,4>*>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++) {
                                sampler.outputsVec4.push_back(buf[index]);
                            }
                            break;
                        }
                        default: {
                            std::cout << "unknown type" << std::endl;
                            break;
                        }
                    }
                }

                animation.samplers.push_back(sampler);
            }

            // Channels
            for (const auto &source: anim.channels)
            {
                Animation::AnimationChannel channel{};

                if (source.target_path == "rotation") {
                    channel.path = Animation::AnimationChannel::PathType::ROTATION;
                }
                if (source.target_path == "translation") {
                    channel.path = Animation::AnimationChannel::PathType::TRANSLATION;
                }
                if (source.target_path == "scale") {
                    channel.path = Animation::AnimationChannel::PathType::SCALE;
                }
                if (source.target_path == "weights") {
                    std::cout << "weights not yet supported, skipping channel" << std::endl;
                    continue;
                }
                channel.samplerIndex = source.sampler;
                if (auto it = instance.nodes.find(source.target_node); it != instance.nodes.end()) {
                    channel.node = &it->second;
                    animation.channels.push_back(channel);
                }
            }

            instance.animations.push_back(animation);
        }
    }
}

void GltfModel::updateAnimation(uint32_t frameIndex, uint32_t index, float time)
{
    if (instances[frameIndex].animations.empty()) {
        std::cout << ".glTF does not contain animation." << std::endl;
        return;
    }
    if (index > static_cast<uint32_t>(instances[frameIndex].animations.size()) - 1) {
        std::cout << "No animation with index " << index << std::endl;
        return;
    }
    Animation &animation = instances[frameIndex].animations[index];

    bool updated = false;
    for (const auto& channel : animation.channels) {
        const Animation::AnimationSampler &sampler = animation.samplers[channel.samplerIndex];
        if (sampler.inputs.size() > sampler.outputsVec4.size()) {
            continue;
        }

        for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
            if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1])) {
                float u = std::max(0.0f, time - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            moon::math::Vector<float,4> trans = mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->translation = moon::math::Vector<float,3>(trans[0],trans[1],trans[2]);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            moon::math::Vector<float,4> trans = mix(sampler.outputsVec4[i], sampler.outputsVec4[i + 1], u);
                            channel.node->scale = moon::math::Vector<float,3>(trans[0],trans[1],trans[2]);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
                            moon::math::Quaternion<float> q1(sampler.outputsVec4[i + 0][3], {sampler.outputsVec4[i + 0][0], sampler.outputsVec4[i + 0][1], sampler.outputsVec4[i + 0][2]});
                            moon::math::Quaternion<float> q2(sampler.outputsVec4[i + 1][3], {sampler.outputsVec4[i + 1][0], sampler.outputsVec4[i + 1][1], sampler.outputsVec4[i + 1][2]});

                            channel.node->rotation = normalize(slerp(q1, q2, u));
                            break;
                        }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto & [_, node] : instances[frameIndex].nodes) {
            node.update();
        }
    }
}

void GltfModel::changeAnimation(uint32_t frameIndex, uint32_t oldIndex, uint32_t newIndex, float startTime, float time, float changeAnimationTime)
{
    if (instances[frameIndex].animations.empty()) {
        std::cout << ".glTF does not contain animation." << std::endl;
        return;
    }
    if (oldIndex > static_cast<uint32_t>(instances[frameIndex].animations.size()) - 1) {
        std::cout << "No animation with index " << oldIndex << std::endl;
        return;
    }
    if (newIndex > static_cast<uint32_t>(instances[frameIndex].animations.size()) - 1) {
        std::cout << "No animation with index " << newIndex << std::endl;
        return;
    }

    Animation &animationOld = instances[frameIndex].animations[oldIndex];
    Animation &animationNew = instances[frameIndex].animations[newIndex];

    bool updated = false;
    for (auto& channel : animationOld.channels) {
        const Animation::AnimationSampler &samplerOld = animationOld.samplers[channel.samplerIndex];
        const Animation::AnimationSampler &samplerNew = animationNew.samplers[channel.samplerIndex];
        if (samplerOld.inputs.size() > samplerOld.outputsVec4.size())
            continue;

        for (size_t i = 0; i < samplerOld.inputs.size(); i++) {
            if ((startTime >= samplerOld.inputs[i]) && (time <= samplerOld.inputs[i]+changeAnimationTime)) {
                float u = std::max(0.0f, time - startTime) / changeAnimationTime;
                if (u <= 1.0f) {
                    switch (channel.path) {
                        case Animation::AnimationChannel::PathType::TRANSLATION: {
                            moon::math::Vector<float,4> trans = mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->translation = moon::math::Vector<float,3>(trans[0],trans[1],trans[2]);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::SCALE: {
                            moon::math::Vector<float,4> trans = mix(samplerOld.outputsVec4[i], samplerNew.outputsVec4[0], u);
                            channel.node->scale = moon::math::Vector<float,3>(trans[0],trans[1],trans[2]);
                            break;
                        }
                        case Animation::AnimationChannel::PathType::ROTATION: {
                            moon::math::Quaternion<float> q1(samplerOld.outputsVec4[i + 0][3], {samplerOld.outputsVec4[i + 0][0], samplerOld.outputsVec4[i + 0][1], samplerOld.outputsVec4[i + 0][2]});
                            moon::math::Quaternion<float> q2(samplerNew.outputsVec4[0][3], {samplerNew.outputsVec4[0][0], samplerNew.outputsVec4[0][1], samplerNew.outputsVec4[0][2]});
                            channel.node->rotation = normalize(slerp(q1, q2, u));
                            break;
                        }
                    }
                    updated = true;
                }
            }
        }
    }
    if (updated) {
        for (auto& [_, node] : instances[frameIndex].nodes) {
            node.update();
        }
    }
}

}
