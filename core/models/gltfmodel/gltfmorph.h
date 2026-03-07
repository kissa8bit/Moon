#ifndef MOON_MODELS_GLTFMODEL_MORPH_H
#define MOON_MODELS_GLTFMODEL_MORPH_H

#include <unordered_map>

#include <utils/memory.h>

#include <interfaces/model.h>

#include "node.h"

namespace moon::models {

struct GltfMorphWeight : public interfaces::MorphWeights {
    GltfMorphWeight() = default;
    GltfMorphWeight(const utils::PhysicalDevice& device, uint32_t morphTargetCount) {
        deviceBuffer = utils::vkDefault::Buffer(
            device, device.device(), sizeof(Buffer),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(deviceBuffer,
            std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", GltfMorphWeight::GltfMorphWeight, deviceBuffer");
        hostBuffer.count = morphTargetCount;
    }

    void update(const Node& node) {
        const uint32_t count = std::min(static_cast<uint32_t>(node.weights.size()), maxTargets);
        hostBuffer.count = count;
        for (uint32_t i = 0; i < count; i++) {
            hostBuffer.weights[i] = node.weights[i];
        }
        deviceBuffer.copy(&hostBuffer);
    }
};

using GltfMorphWeightsMap = std::unordered_map<Node::Id, GltfMorphWeight>;

} // moon::models

#endif // MOON_MODELS_GLTFMODEL_MORPH_H
