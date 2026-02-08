#include "graphics.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

#include <interfaces/object.h>
#include <interfaces/model.h>

namespace moon::deferredGraphics {

Graphics::OutliningExtension::OutliningExtension(const Graphics::Base& parent) : parent(parent) {}

void Graphics::OutliningExtension::create(interfaces::ObjectType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
    type = type | interfaces::ObjectType::outlining;

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parent.parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    const auto bindingDescription = interfaces::VertexInputBindingDescriptionFromObjectType(type);
    const auto attributeDescriptions = interfaces::AttributeDescriptionsFromObjectType(type);

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
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    auto& pipelineDesc = pipelineDescs[type];

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_SHADER_STAGE_ALL;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(interfaces::Material::Buffer);
    std::vector<VkDescriptorSetLayout> descriptorSetLayout = {
        parent.descriptorSetLayout,
        parent.objectDescriptorSetLayout,
        parent.skeletonDescriptorSetLayout,
        parent.materialDescriptorSetLayout
    };
    pipelineDesc.pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayout, pushConstantRange);

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
        pipelineInfo.back().layout = pipelineDesc.pipelineLayout;
        pipelineInfo.back().renderPass = renderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipelineDesc.pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void Graphics::OutliningExtension::render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const
{
    if (!parent.objects) return;

    for(const auto& object: *parent.objects){
        if (!object) continue;

        const auto mask = object->objectMask();
        const auto property = mask.property();
        const auto type = mask.type();
        const auto model = object->model();

        if (!model) continue;
        if (!property.has(interfaces::ObjectProperty::enable)) continue;
        if (!type.has_any(interfaces::ObjectType::Value::baseTypes)) continue;
        if (!type.has(interfaces::ObjectType::Value::outlining)) continue;

        const utils::vkDefault::DescriptorSets descriptorSets = { parent.descriptorSets.at(frameNumber), object->getDescriptorSet(frameNumber) };

        const auto& pipelineDesc = pipelineDescs.at(type);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineDesc.pipeline);

        uint32_t primirives = 0;
        model->render(object->getInstanceNumber(frameNumber), commandBuffer, pipelineDesc.pipelineLayout, descriptorSets, primirives);
    }
}

}
