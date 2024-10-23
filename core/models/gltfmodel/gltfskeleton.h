#ifndef GLTFSKELETON_H
#define GLTFSKELETON_H

#include <unordered_map>

#include "skin.h"
#include "model.h"
#include "gltfutils.h"

namespace moon::models {

struct GltfSkeleton : public interfaces::Skeleton {
    const Skin* skin{ nullptr };

    GltfSkeleton() = default;
    GltfSkeleton(const utils::PhysicalDevice& device, const Skin* skin = nullptr) : skin(skin) {
        const size_t jointCount = skin ? skin->size() : 0;
        const size_t bufferSize = sizeof(math::mat4) * (jointCount + 1);
        deviceBuffer = utils::vkDefault::Buffer(device, device.device(), bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        moon::utils::Memory::instance().nameMemory(deviceBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", GltfSkeleton::GltfSkeleton, deviceBuffer");
    }

    void update(const Nodes& nodeMap, Node::Id rootNode) {
        hostBuffer.matrix = transpose(nodeMap.at(rootNode)->matrix);
        if (skin) {
            const auto& joints = (*skin);
            for (size_t i = 0; i < joints.size(); i++) {
                hostBuffer.jointMatrix[i] = transpose(inverse(hostBuffer.matrix) * nodeMap.at(joints[i].jointedNodeId)->matrix * joints[i].inverseBindMatrices);
            }
        }
        deviceBuffer.copy(&hostBuffer);
    }
};

using GltfSkeletons = std::unordered_map<Node::Id, GltfSkeleton>;

}

#endif