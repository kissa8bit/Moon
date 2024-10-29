#include "postProcessing.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::workflows{

PostProcessingGraphics::PostProcessingGraphics(PostProcessingParameters& parameters) : parameters(parameters), postProcessing(parameters) {}

void PostProcessingGraphics::createAttachments(utils::AttachmentsDatabase& aDatabase){
    utils::createAttachments(physicalDevice, device, parameters.imageInfo, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.postProcessing, parameters.enable, &frame);
}

void PostProcessingGraphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(parameters.imageInfo.Format)
    };

    utils::vkDefault::SubpassInfos subpassInfos = utils::vkDefault::subpassInfos(attachments.size());

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, dependencies);
}

void PostProcessingGraphics::createFramebuffers()
{
    framebuffers.resize(parameters.imageInfo.Count);
    for (size_t i = 0; i < framebuffers.size(); i++) {
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

void PostProcessingGraphics::PostProcessing::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

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

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {utils::vkDefault::colorBlendAttachmentState(VK_FALSE)};
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts);

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

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void PostProcessingGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "postProcessing/postProcessingVert.spv"},
            {workflows::ShaderType::Fragment, "postProcessing/postProcessingFrag.spv"}
        };
        postProcessing.create(shaderNames, device, renderPass);
        created = true;
    }
}

void PostProcessingGraphics::updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase) {
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++){
        VkDescriptorImageInfo layersImageInfo = aDatabase.descriptorImageInfo(parameters.in.baseColor, i);
        VkDescriptorImageInfo blurImageInfo = aDatabase.descriptorImageInfo(parameters.in.blur, i);
        VkDescriptorImageInfo ssaoImageInfo = aDatabase.descriptorImageInfo(parameters.in.ssao, i);
        VkDescriptorImageInfo bbImageInfo = aDatabase.descriptorImageInfo(parameters.in.boundingBox, i);
        VkDescriptorImageInfo bloomImageInfo =  aDatabase.descriptorImageInfo(parameters.in.bloom, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &layersImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &blurImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &ssaoImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = postProcessing.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bbImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void PostProcessingGraphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues;
    clearValues.push_back(VkClearValue{});

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessing.pipelineLayout, 0, 1, &postProcessing.descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

}
