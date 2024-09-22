#include "../graphics.h"
#include "operations.h"
#include "vkdefault.h"
#include "object.h"
#include "model.h"
#include "vector.h"

namespace moon::deferredGraphics {

Graphics::OutliningExtension::OutliningExtension(const Graphics::Base& parent) : parent(parent) {}

void Graphics::OutliningExtension::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    auto bindingDescription = interfaces::Vertex::getBindingDescription();
    auto attributeDescriptions = interfaces::Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkViewport viewport = utils::vkDefault::viewport({0,0}, parent.parameters.imageInfo.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, parent.parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilDisable();

    depthStencil.stencilTestEnable = VK_TRUE;
    depthStencil.back.compareOp = VK_COMPARE_OP_NOT_EQUAL;
    depthStencil.back.failOp = VK_STENCIL_OP_KEEP;
    depthStencil.back.depthFailOp = VK_STENCIL_OP_KEEP;
    depthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
    depthStencil.back.compareMask = 0xff;
    depthStencil.back.writeMask = 0xff;
    depthStencil.back.reference = 1;
    depthStencil.front = depthStencil.back;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(interfaces::MaterialBlock);
    std::vector<VkDescriptorSetLayout> descriptorSetLayout = {
        parent.descriptorSetLayout,
        parent.objectDescriptorSetLayout,
        parent.primitiveDescriptorSetLayout,
        parent.materialDescriptorSetLayout
    };
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
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void Graphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const
{
    if (!parent.objects) return;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for(const auto& object: *parent.objects){
        if (!object) continue;
        if (!object->outlining()) continue;

        const auto pipelineFlagBits = object->pipelineFlagBits();
        const auto model = object->model();

        if (!model) continue;
        if (!(object->getEnable() && (interfaces::ObjectType::base & pipelineFlagBits))) continue;

        VkDeviceSize offsets = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, model->vertexBuffer(), &offsets);
        if (auto indexBuffer = *model->indexBuffer(); indexBuffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        utils::vkDefault::DescriptorSets descriptorSets = { parent.descriptorSets[frameNumber], object->getDescriptorSet(frameNumber) };

        uint32_t primirives = 0;
        model->render(object->getInstanceNumber(frameNumber), commandBuffer, pipelineLayout, descriptorSets, primirives);
    }
}

}
