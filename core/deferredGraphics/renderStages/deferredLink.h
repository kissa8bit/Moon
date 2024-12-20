#ifndef DEFERRED_LINK_H
#define DEFERRED_LINK_H

#include <vector>
#include <filesystem>

#include "linkable.h"
#include "attachments.h"
#include "vkdefault.h"

#include "linearAlgebra.h"

namespace moon::deferredGraphics {

class DeferredLink : public graphicsManager::Linkable
{
private:
    utils::vkDefault::PipelineLayout        pipelineLayout;
    utils::vkDefault::Pipeline              pipeline;
    utils::vkDefault::DescriptorSetLayout   descriptorSetLayout;

    utils::vkDefault::DescriptorPool        descriptorPool;
    utils::vkDefault::DescriptorSets        descriptorSets;

    void createPipeline(VkDevice device, const std::filesystem::path& shadersPath, const utils::vkDefault::ImageInfo& info);
    void createDescriptors(VkDevice device, const utils::vkDefault::ImageInfo& info, const utils::Attachments* attachment);

public:
    DeferredLink() = default;
    DeferredLink(VkDevice device, const std::filesystem::path& shadersPath, const utils::vkDefault::ImageInfo& info, VkRenderPass renderPass, const graphicsManager::PositionInWindow& position, const utils::Attachments* attachment);
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;
};

}
#endif // DEFERRED_LINK_H
