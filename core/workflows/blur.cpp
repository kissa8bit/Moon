#include "blur.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::workflows {

GaussianBlur::GaussianBlur(GaussianBlurParameters& parameters) : parameters(parameters), xblur(parameters, 0), yblur(parameters, 2) {}

void GaussianBlur::createAttachments(utils::AttachmentsDatabase& aDatabase)
{
    utils::createAttachments(physicalDevice, device, parameters.imageInfo, 1, &bufferAttachment, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    utils::createAttachments(physicalDevice, device, parameters.imageInfo, 1, &frame, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    aDatabase.addAttachmentData(parameters.out.blur, parameters.enable, &frame);
}

void GaussianBlur::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(parameters.imageInfo.Format),
        utils::Attachments::imageDescription(parameters.imageInfo.Format)
    };

    utils::vkDefault::SubpassInfos subpassInfos;
    auto& xblur = subpassInfos.emplace_back();
    xblur.out = {
        VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        VkAttachmentReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };
    auto& imageLayoutTransition = subpassInfos.emplace_back();
    imageLayoutTransition.out = {
        VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };
    imageLayoutTransition.in = {
        VkAttachmentReference{1,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
    };
    auto& yblur = subpassInfos.emplace_back();
    yblur.out = {
        VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = 0;
        dependencies.back().dstSubpass = 1;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().srcAccessMask = 0;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies.back().dstAccessMask = 0;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = 1;
        dependencies.back().dstSubpass = 2;
        dependencies.back().srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().srcAccessMask = 0;
        dependencies.back().dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, dependencies);
}

void GaussianBlur::createFramebuffers(){
    framebuffers.resize(parameters.imageInfo.Count);
    for (uint32_t i = 0; i < static_cast<uint32_t>(framebuffers.size()); i++) {
        std::vector<VkImageView> attachments = { frame.imageView(i), bufferAttachment.imageView(i) };
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void GaussianBlur::Blur::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
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
    if(subpassNumber == 0){
        colorBlendAttachment.push_back(utils::vkDefault::colorBlendAttachmentState(VK_FALSE));
    }
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(float);
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
        pipelineInfo.back().subpass = subpassNumber;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout}, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void GaussianBlur::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabasep) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createAttachments(aDatabasep);
        createRenderPass();
        createFramebuffers();
        const workflows::ShaderNames xShaderNames = {
            {workflows::ShaderType::Vertex, "gaussianBlur/xBlurVert.spv"},
            {workflows::ShaderType::Fragment, "gaussianBlur/xBlurFrag.spv"}
        };
        xblur.create(xShaderNames, device, renderPass);
        const workflows::ShaderNames yShaderNames = {
            {workflows::ShaderType::Vertex, "gaussianBlur/yBlurVert.spv"},
            {workflows::ShaderType::Fragment, "gaussianBlur/yBlurFrag.spv"}
        };
        yblur.create(yShaderNames, device, renderPass);
        created = true;
    }
}

void GaussianBlur::updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase) {
    if(!parameters.enable || !created) return;

    auto updateDescriptorSets = [](VkDevice device, const utils::Attachments& image, const utils::vkDefault::DescriptorSets& descriptorSets) {
        for (uint32_t i = 0; i < image.count(); i++) {
            utils::descriptorSet::Writes writes;
            WRITE_DESCRIPTOR(writes, descriptorSets[i], image.descriptorImageInfo(i));
            utils::descriptorSet::update(device, writes);
        }
    };

    updateDescriptorSets(device, *aDatabase.get(parameters.in.blur), xblur.descriptorSets);
    updateDescriptorSets(device, bufferAttachment, yblur.descriptorSets);
}

void GaussianBlur::updateCommandBuffer(uint32_t frameNumber){
    if(!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues = { frame.clearValue() , bufferAttachment.clearValue()};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(commandBuffers[frameNumber], xblur.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &parameters.blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, xblur.pipelineLayout, 0, 1, &xblur.descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);
    vkCmdNextSubpass(commandBuffers[frameNumber], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(commandBuffers[frameNumber], yblur.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(float), &parameters.blurDepth);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, yblur.pipelineLayout, 0, 1, &yblur.descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

} // moon::workflows
