#include "rayTracingLink.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::rayTracingGraphics {

RayTracingLink::RayTracingLink(VkDevice device, const RayTracingLinkParameters& parameters, VkRenderPass renderPass, const moon::utils::AttachmentsDatabase& aDatabase)
    : parameters(parameters) {
    createPipeline(device, renderPass);
    createDescriptors(device, aDatabase);
};

void RayTracingLink::createPipeline(VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(moon::utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / "linkable/linkableVert.spv");
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parameters.shadersPath / "linkable/linkableFrag.spv");
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, parameters.imageInfo.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

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
    pipelineInfo.back().renderPass = renderPass;
    pipelineInfo.back().subpass = 0;
    pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void RayTracingLink::createDescriptors(VkDevice device, const moon::utils::AttachmentsDatabase& aDatabase) {
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);

    for (size_t i = 0; i < parameters.imageInfo.Count; i++) {
        auto descriptorSet = descriptorSets[i];
        const auto imageInfo = aDatabase.descriptorImageInfo(parameters.in.color, i);
        const auto bloomImageInfo = aDatabase.descriptorImageInfo(parameters.in.bloom, i);
        const auto bbImageInfo = aDatabase.descriptorImageInfo(parameters.in.boundingBox, i);

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, descriptorSet, imageInfo);
        utils::descriptorSet::write(writes, descriptorSet, bloomImageInfo);
        utils::descriptorSet::write(writes, descriptorSet, bbImageInfo);
        utils::descriptorSet::update(device, writes);
    }
}

void RayTracingLink::draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const
{
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(position), &position);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageNumber], 0, nullptr);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);
}

}
