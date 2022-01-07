#include "graphics.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::deferredGraphics {

void Graphics::Lighting::createPipeline(uint8_t mask, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = utils::vkDefault::viewport({0,0}, parameters.imageInfo.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = utils::vkDefault::vertexInputState();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilDisable();

    VkPipelineColorBlendAttachmentState customBlendAttachment{};
        customBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        customBlendAttachment.blendEnable = VK_TRUE;
        customBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        customBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        customBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment ={
        utils::vkDefault::colorBlendAttachmentState(VK_TRUE),
        customBlendAttachment,
        utils::vkDefault::colorBlendAttachmentState(VK_TRUE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        shadowDescriptorSetLayout,
        lightDescriptorSetLayoutMap[mask]
    };
    pipelineLayoutMap[mask] = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts);

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
        pipelineInfo.back().layout = pipelineLayoutMap[mask];
        pipelineInfo.back().renderPass = renderPass;
        pipelineInfo.back().subpass = 1;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelineMap[mask] = utils::vkDefault::Pipeline(device, pipelineInfo);
}

}
