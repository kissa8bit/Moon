#ifndef MOON_MODELS_GLTFSKELETON_H
#define MOON_MODELS_GLTFSKELETON_H

#include <utils/memory.h>

#include "skin.h"
#include "gltfutils.h"

namespace moon::models {

struct GltfSkeleton : public interfaces::Skeleton {
    const Skin* skin{ nullptr };

    GltfSkeleton() = default;
    GltfSkeleton(const utils::PhysicalDevice& device, const Skin* skin = nullptr) : skin(skin) {
        const size_t jointCount = skin ? skin->joints.size() : 0;
        const size_t bufferSize = sizeof(math::mat4) * (jointCount + 1);
        deviceBuffer = utils::vkDefault::Buffer(device, device.device(), bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(deviceBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", GltfSkeleton::GltfSkeleton, deviceBuffer");
    }

    void update(const Nodes& nodes, Node::Id rootNode) {
        hostBuffer.matrix = transpose(nodes.at(rootNode).matrix);
        if (skin) {
            const auto& joints = skin->joints;
            for (size_t i = 0; i < joints.size(); i++) {
                hostBuffer.jointMatrix[i] = transpose(inverse(hostBuffer.matrix) * nodes.at(joints[i].jointedNodeId).matrix * joints[i].inverseBindMatrices);
            }
        }
        deviceBuffer.copy(&hostBuffer);
    }
};

using GltfSkeletons = std::unordered_map<Node::Id, GltfSkeleton>;

} // moon::models

#endif // MOON_MODELS_GLTFSKELETON_H