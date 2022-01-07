#include "deferredLink.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::deferredGraphics {

DeferredLink::DeferredLink(VkDevice device, const std::filesystem::path& shadersPath, const utils::vkDefault::ImageInfo& info, VkRenderPass renderPass, const graphicsManager::PositionInWindow& position, const utils::Attachments* attachment)
    : Linkable(renderPass, position) {
    createPipeline(device, shadersPath, info);
    createDescriptors(device, info, attachment);
}

void DeferredLink::createPipeline(VkDevice device, const std::filesystem::path& shadersPath, const utils::vkDefault::ImageInfo& info) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, shadersPath / "linkable/deferredLinkableVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, shadersPath / "linkable/deferredLinkableFrag.spv");
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = utils::vkDefault::viewport({0,0}, info.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, info.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(graphicsManager::PositionInWindow);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfo;
    pipelineInfo.push_back(VkGraphicsPipelineCreateInfo{});
        pipelineInfo.back().sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.back().pNext = nullptr;
        pipelineInfo.back().stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.back().pStages = shaderStages.data();
        pipelineInfo.back().pVertexInputState = &vertexInputInfo;
        pipelineInfo.back().pInputAssemblyState = &inputAssembly;
        pipelineInfo.back().pViewportState = &viewportState;
        pipelineInfo.back().pRasterizationState = &rasterizer;
        pipelineInfo.back().pMultisampleState = &multisampling;
        pipelineInfo.back().pColorBlendState = &colorBlending;
        pipelineInfo.back().layout = pipelineLayout;
        pipelineInfo.back().renderPass = renderPass();
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void DeferredLink::createDescriptors(VkDevice device, const utils::vkDefault::ImageInfo& info, const utils::Attachments* attachment) {
    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout}, info.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, info.Count);

    if (!CHECK_M(attachment, std::string("[ Link::createDescriptors ] attachment is nullptr"))) return;

    for (size_t i = 0; i < info.Count; i++) {
        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSets.at(i), attachment->descriptorImageInfo(i));
        utils::descriptorSet::update(device, writes);
    }
}

void DeferredLink::draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const {
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(position), &position);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets.at(imageNumber), 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
}

} // moon::deferredGraphics
