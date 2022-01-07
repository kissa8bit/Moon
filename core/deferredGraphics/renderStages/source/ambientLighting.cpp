#include "../graphics.h"
#include "operations.h"

namespace moon::deferredGraphics {

struct LightPassPushConst{
    alignas(4) float minAmbientFactor;
};

Graphics::AmbientLighting::AmbientLighting(const Lighting& parent) : parent(parent) {}

void Graphics::AmbientLighting::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = utils::vkDefault::viewport({0,0}, parent.parameters.imageInfo.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, parent.parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilDisable();

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment ={
        utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        utils::vkDefault::colorBlendAttachmentState(VK_TRUE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(LightPassPushConst);
    std::vector<VkDescriptorSetLayout> descriptorSetLayout = {parent.descriptorSetLayout};
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayout, pushConstantRange);

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
        pipelineInfo.back().subpass = 1;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void Graphics::AmbientLighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const {
    LightPassPushConst pushConst{ parent.parameters.minAmbientFactor };
    vkCmdPushConstants(commandBuffers, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(LightPassPushConst), &pushConst);

    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &parent.descriptorSets[frameNumber], 0, nullptr);
    vkCmdDraw(commandBuffers, 6, 1, 0, 0);
}

}
