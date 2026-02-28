#include "scattering.h"

#include <utils/operations.h>
#include <utils/depthMap.h>

namespace moon::workflows {

struct ScatteringPushConst {
    alignas(4) uint32_t width { 0 };
    alignas(4) uint32_t height { 0 };
};

Scattering::Scattering(ScatteringParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps) :
    parameters(parameters), lighting(parameters, lightSources, depthMaps)
{}

void Scattering::createAttachments(utils::AttachmentsDatabase& aDatabase){
    const utils::vkDefault::ImageInfo info = { parameters.imageInfo.Count, VK_FORMAT_R32G32B32A32_SFLOAT, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    frame = utils::Attachments(physicalDevice, device, info);
    aDatabase.addAttachmentData(parameters.out.scattering, parameters.enable, &frame);
}

void Scattering::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(frame.format())
    };

    utils::vkDefault::SubpassInfos subpassInfos = utils::vkDefault::subpassInfos(attachments.size());

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
    dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies.back().dstSubpass = 0;
    dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies.back().srcAccessMask = 0;
    dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, dependencies);
}

void Scattering::createFramebuffers()
{
    framebuffers.resize(parameters.imageInfo.Count);
    for(size_t i = 0; i < parameters.imageInfo.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &frame.imageView(i);
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void Scattering::Lighting::createPipeline(interfaces::LightType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
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

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(1,&colorBlendAttachment);

    auto& pipelineDesc = pipelineDescs[type];

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_SHADER_STAGE_ALL;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(ScatteringPushConst);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        shadowDescriptorSetLayout,
        pipelineDesc.descriptorSetLayout = interfaces::Light::createDescriptorSetLayout(device)
    };
    pipelineDesc.pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

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
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelineDesc.pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}

void Scattering::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(!parameters.enable || created) return;

    createAttachments(aDatabase);
    createRenderPass();
    createFramebuffers();

    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    lighting.descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
    lighting.shadowDescriptorSetLayout = utils::DepthMap::createDescriptorSetLayout(device);

    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "scattering/scatteringSpotCircleVert.spv"},
            {workflows::ShaderType::Fragment, "scattering/scatteringSpotCircleFrag.spv"}
        };
        lighting.createPipeline(interfaces::LightType::spotCircle, shaderNames, device, renderPass);
    }
    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "scattering/scatteringSpotSquareVert.spv"},
            {workflows::ShaderType::Fragment, "scattering/scatteringSpotSquareFrag.spv"}
        };
        lighting.createPipeline(interfaces::LightType::spotSquare, shaderNames, device, renderPass);
    }

    lighting.descriptorPool = utils::vkDefault::DescriptorPool(device, { &lighting.descriptorSetLayout }, parameters.imageInfo.Count);
    lighting.descriptorSets = lighting.descriptorPool.allocateDescriptorSets(lighting.descriptorSetLayout, parameters.imageInfo.Count);

    created = true;
}

void Scattering::updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) {
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++)
    {
        auto descriptorSet = lighting.descriptorSets[i];

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSet, bDatabase.descriptorBufferInfo(parameters.in.camera, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.depth, i));
        utils::descriptorSet::update(device, writes);
    }
}

void Scattering::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;
	if (!lighting.lightSources || !lighting.depthMaps) return;

    std::vector<VkClearValue> clearValues = {frame.clearValue()};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    for(auto& lightSource: *lighting.lightSources){
        if (!lightSource) continue;

        const auto mask = lightSource->lightMask();
        const auto type = mask.type();
        const auto property = mask.property();

        if(!property.has(interfaces::LightProperty::enableScattering)) continue;
        if(lighting.pipelineDescs.find(type) == lighting.pipelineDescs.end()) continue;
        if (lighting.depthMaps->find(lightSource) == lighting.depthMaps->end()) continue;

        const auto& pipelineDesc = lighting.pipelineDescs.at(type);
        const auto& depthMap = lighting.depthMaps->at(lightSource);

        ScatteringPushConst pushConst{ parameters.imageInfo.Extent.width, parameters.imageInfo.Extent.height};
        vkCmdPushConstants(commandBuffers[frameNumber], pipelineDesc.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(ScatteringPushConst), &pushConst);

        const utils::vkDefault::DescriptorSets descriptors = { lighting.descriptorSets.at(frameNumber), depthMap.descriptorSets().at(frameNumber) };
        lightSource->render(frameNumber, commandBuffers.at(frameNumber), descriptors, pipelineDesc.pipelineLayout, pipelineDesc.pipeline);
    }

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

} // moon::workflows
