#ifndef MOON_RAY_TRACING_GRAPHICS_RAYTRACINGLINK_H
#define MOON_RAY_TRACING_GRAPHICS_RAYTRACINGLINK_H

#include <vector>
#include <filesystem>

#include <utils/attachments.h>

#include <math/linearAlgebra.h>

#include <graphicsManager/graphicsInterface.h>

namespace moon::rayTracingGraphics {

struct RayTracingLinkParameters{
    struct{
        std::string color;
        std::string bloom;
        std::string boundingBox;
    }in;
    struct{}out;
    std::filesystem::path shadersPath;
    moon::utils::vkDefault::ImageInfo imageInfo;
};

class RayTracingLink {
private:
    RayTracingLinkParameters                parameters;

    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;

    graphicsManager::PositionInWindow       position;

    void createPipeline(VkDevice device, VkRenderPass renderPass);
    void createDescriptors(VkDevice device, const moon::utils::AttachmentsDatabase& aDatabase);

public:
    RayTracingLink() = default;
    RayTracingLink(VkDevice device, const RayTracingLinkParameters& parameters, VkRenderPass renderPass, const moon::utils::AttachmentsDatabase& aDatabase);
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const;
};

} // moon::rayTracingGraphics

#endif MOON_RAY_TRACING_GRAPHICS_RAYTRACINGLINK_H
