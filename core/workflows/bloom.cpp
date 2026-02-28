#include "bloom.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::workflows {

struct BloomPushConst{
    alignas (4) float deltax;
    alignas (4) float deltay;
    alignas (4) float blitFactor;
};

BloomGraphics::BloomGraphics(BloomParameters& parameters) 
    : parameters(parameters), filter(parameters), bloom(parameters)
{}

void BloomGraphics::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(bufferAttachment.format(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
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

void BloomGraphics::createFramebuffers(){
    framebuffers.resize(parameters.imageInfo.Count * (frames.size() + 1));
    for(size_t i = 0; i < frames.size(); i++){
        for (size_t j = 0; j < parameters.imageInfo.Count; j++){
            VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = &frames[i].imageView(j);
                framebufferInfo.width = parameters.imageInfo.Extent.width;
                framebufferInfo.height = parameters.imageInfo.Extent.height;
                framebufferInfo.layers = 1;
            framebuffers[parameters.imageInfo.Count * i + j] = utils::vkDefault::Framebuffer(device, framebufferInfo);
        }
    }
    for(size_t i = 0; i < parameters.imageInfo.Count; i++){
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &bufferAttachment.imageView(i);
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[parameters.imageInfo.Count * frames.size() + i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void BloomGraphics::Filter::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
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

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(BloomPushConst);
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

    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void BloomGraphics::Bloom::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.blitAttachmentsCount));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    uint32_t specializationData = parameters.blitAttachmentsCount;
    VkSpecializationMapEntry specializationMapEntry{};
    specializationMapEntry.constantID = 0;
    specializationMapEntry.offset = 0;
    specializationMapEntry.size = sizeof(uint32_t);
    VkSpecializationInfo specializationInfo;
    specializationInfo.mapEntryCount = 1;
    specializationInfo.pMapEntries = &specializationMapEntry;
    specializationInfo.dataSize = sizeof(specializationData);
    specializationInfo.pData = &specializationData;

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment), specializationInfo);
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

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(BloomPushConst);
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

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void BloomGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        frames.resize(parameters.blitAttachmentsCount);
        const utils::vkDefault::ImageInfo u8Info = { parameters.imageInfo.Count, VK_FORMAT_R8G8B8A8_UNORM, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
        utils::createAttachments(physicalDevice, device, u8Info, parameters.blitAttachmentsCount, frames.data(), VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        bufferAttachment = utils::Attachments(physicalDevice, device, u8Info, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &bufferAttachment);

        createRenderPass();
        createFramebuffers();
        const workflows::ShaderNames filterShaderNames = {
            {workflows::ShaderType::Vertex, "customFilter/customFilterVert.spv"},
            {workflows::ShaderType::Fragment, "customFilter/customFilterFrag.spv"}
        };
        filter.create(filterShaderNames, device, renderPass);
        const workflows::ShaderNames bloomShaderNames = {
            {workflows::ShaderType::Vertex, "bloom/bloomVert.spv"},
            {workflows::ShaderType::Fragment, "bloom/bloomFrag.spv"}
        };
        bloom.create(bloomShaderNames, device, renderPass);
        created = true;
    }
}

void BloomGraphics::updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase)
{
    if(!parameters.enable || !created) return;

    srcAttachment = aDatabase.get(parameters.in.bloom);

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++) {
        const auto imageInfo = bufferAttachment.descriptorImageInfo(i);

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, filter.descriptorSets[i], imageInfo);
        utils::descriptorSet::update(device, writes);
    }

    for (size_t i = 0; i < parameters.imageInfo.Count; i++) {
        std::vector<VkDescriptorImageInfo> imageInfos(parameters.blitAttachmentsCount);
        for(uint32_t j = 0; j < imageInfos.size(); j++){
            imageInfos[j] = frames[j].descriptorImageInfo(i);
        }

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, bloom.descriptorSets[i], imageInfos);
        utils::descriptorSet::update(device, writes);
    }
}

void BloomGraphics::updateCommandBuffer(uint32_t frameNumber){
    if(!parameters.enable || !created) return;

    VkImageSubresourceRange ImageSubresourceRange{};
        ImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ImageSubresourceRange.baseMipLevel = 0;
        ImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        ImageSubresourceRange.baseArrayLayer = 0;
        ImageSubresourceRange.layerCount = 1;
    VkClearColorValue clearColorValue{};
        clearColorValue.uint32[0] = 0;
        clearColorValue.uint32[1] = 0;
        clearColorValue.uint32[2] = 0;
        clearColorValue.uint32[3] = 0;

    std::vector<VkImage> blitImages(frames.size());
    blitImages[0] = srcAttachment->image(frameNumber);
    utils::texture::transitionLayout(commandBuffers[frameNumber], blitImages[0], parameters.inputImageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

    for(size_t i = 1; i < frames.size(); i++){
        blitImages[i] = frames[i - 1].image(frameNumber);
    }

    VkImage blitBufferImage = bufferAttachment.image(frameNumber);
    uint32_t width = parameters.imageInfo.Extent.width;
    uint32_t height = parameters.imageInfo.Extent.height;

    for(uint32_t k = 0; k < frames.size(); k++){
        utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, (k == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
        vkCmdClearColorImage(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColorValue, 1, &ImageSubresourceRange);
        utils::texture::blitDown(commandBuffers[frameNumber], blitImages[k], 0, blitBufferImage, 0, width, height, 0, 1, parameters.blitFactor);
        utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

        render(commandBuffers[frameNumber], frames[k], frameNumber, k * parameters.imageInfo.Count + frameNumber, &filter);
    }
    for(uint32_t k = 0; k < frames.size(); k++){
        utils::texture::transitionLayout(commandBuffers[frameNumber],frames[k].image(frameNumber), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    }
    render(commandBuffers[frameNumber], bufferAttachment, frameNumber, static_cast<uint32_t>(frames.size()) * parameters.imageInfo.Count + frameNumber, &bloom);
    utils::texture::transitionLayout(commandBuffers[frameNumber], blitBufferImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
}

void BloomGraphics::render(VkCommandBuffer commandBuffer, const utils::Attachments& image, uint32_t frameNumber, uint32_t framebufferIndex, Workbody* worker)
{
    std::vector<VkClearValue> clearValues = { image.clearValue() };

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[framebufferIndex];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        BloomPushConst pushConst{ parameters.xSamplerStep, parameters.ySamplerStep, parameters.blitFactor};
        vkCmdPushConstants(commandBuffer, worker->pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(BloomPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worker->pipelineLayout, 0, 1, &worker->descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
}

} // moon::workflows
