#include "layersCombiner.h"
#include "operations.h"
#include "vkdefault.h"

namespace moon::deferredGraphics {

struct LayersCombinerPushConst {
    alignas(4) int enableScatteringRefraction{ true };
    alignas(4) int enableTransparentLayers{ true };
    alignas(4) float blurDepth{ 1.0f };
};

LayersCombiner::LayersCombiner(LayersCombinerParameters& parameters) : combiner(parameters), parameters(parameters) {}

void LayersCombiner::createAttachments(utils::AttachmentsDatabase& aDatabase)
{
    auto createAttachments = [](VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo image, uint32_t attachmentsCount, utils::Attachments* pAttachments){
        for(size_t index=0; index < attachmentsCount; index++){
            pAttachments[index] = utils::Attachments(physicalDevice, device, image, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | (index==1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0));
        }
    };

    createAttachments(physicalDevice, device, parameters.imageInfo, LayersCombinerAttachments::size(), &frame);
    aDatabase.addAttachmentData(parameters.out.color, parameters.enable, &frame.color);
    aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &frame.bloom);
    aDatabase.addAttachmentData(parameters.out.blur, parameters.enable, &frame.blur);
}

void LayersCombiner::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(parameters.imageInfo.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        utils::Attachments::imageDescription(parameters.imageInfo.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        utils::Attachments::imageDescription(parameters.imageInfo.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
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

void LayersCombiner::createFramebuffers(){
    framebuffers.resize(parameters.imageInfo.Count);
    for(size_t i = 0; i < parameters.imageInfo.Count; i++){
        std::vector<VkImageView> attachments = { frame.color.imageView(i), frame.bloom.imageView(i), frame.blur.imageView(i) };
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

void LayersCombiner::Combiner::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::bufferFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.transparentLayersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.transparentLayersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.transparentLayersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.transparentLayersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), parameters.transparentLayersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    uint32_t specializationData = parameters.transparentLayersCount;
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

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment(LayersCombinerAttachments::size(), utils::vkDefault::colorBlendAttachmentState(VK_FALSE));
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(LayersCombinerPushConst);
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

    descriptorPool = utils::vkDefault::DescriptorPool(device, {&descriptorSetLayout}, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void LayersCombiner::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "layersCombiner/layersCombinerVert.spv"},
            {workflows::ShaderType::Fragment, "layersCombiner/layersCombinerFrag.spv"}
        };
        combiner.create(shaderNames, device, renderPass);
        created = true;
    }
}

void LayersCombiner::updateDescriptors(
    const utils::BuffersDatabase& bDatabase,
    const utils::AttachmentsDatabase& aDatabase) {
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++)
    {
        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);
        VkDescriptorImageInfo colorImageInfo = aDatabase.descriptorImageInfo(parameters.in.color, i);
        VkDescriptorImageInfo bloomImageInfo = aDatabase.descriptorImageInfo(parameters.in.bloom, i);
        VkDescriptorImageInfo positionImageInfo = aDatabase.descriptorImageInfo(parameters.in.position, i);
        VkDescriptorImageInfo normalImageInfo = aDatabase.descriptorImageInfo(parameters.in.normal, i);
        VkDescriptorImageInfo depthImageInfo = aDatabase.descriptorImageInfo(parameters.in.depth, i, parameters.in.defaultDepthTexture);
        VkDescriptorImageInfo skyboxImageInfo = aDatabase.descriptorImageInfo(parameters.in.skyboxColor, i);
        VkDescriptorImageInfo skyboxBloomImageInfo = aDatabase.descriptorImageInfo(parameters.in.skyboxBloom, i);
        VkDescriptorImageInfo scatteringImageInfo = aDatabase.descriptorImageInfo(parameters.in.scattering, i);
        VkDescriptorImageInfo sslrImageInfo = aDatabase.descriptorImageInfo(parameters.in.sslr, i);

        std::vector<VkDescriptorImageInfo> colorLayersImageInfo(parameters.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> bloomLayersImageInfo(parameters.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> positionLayersImageInfo(parameters.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> normalLayersImageInfo(parameters.transparentLayersCount);
        std::vector<VkDescriptorImageInfo> depthLayersImageInfo(parameters.transparentLayersCount);

        for(uint32_t index = 0; index < parameters.transparentLayersCount; index++){
            std::string key = parameters.in.transparency + std::to_string(index) + ".";

            colorLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.color, i);
            bloomLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.bloom, i);
            positionLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.position, i);
            normalLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.normal, i);
            depthLayersImageInfo[index] = aDatabase.descriptorImageInfo(key + parameters.in.depth, i, parameters.in.defaultDepthTexture);
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(static_cast<uint32_t>(descriptorWrites.size() - 1));
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(static_cast<uint32_t>(descriptorWrites.size() - 1));
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &colorImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &bloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &positionImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &normalImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &depthImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = parameters.transparentLayersCount;
            descriptorWrites.back().pImageInfo = colorLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = parameters.transparentLayersCount;
            descriptorWrites.back().pImageInfo = bloomLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = parameters.transparentLayersCount;
            descriptorWrites.back().pImageInfo = positionLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = parameters.transparentLayersCount;
            descriptorWrites.back().pImageInfo = normalLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = parameters.transparentLayersCount;
            descriptorWrites.back().pImageInfo = depthLayersImageInfo.data();
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &skyboxBloomImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &scatteringImageInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = combiner.descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &sslrImageInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void LayersCombiner::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues = { frame.color.clearValue(), frame.bloom.clearValue(), frame.blur.clearValue() };

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[frameNumber];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers[frameNumber], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        LayersCombinerPushConst pushConst{};
            pushConst.enableScatteringRefraction = static_cast<int>(parameters.enableScatteringRefraction);
            pushConst.enableTransparentLayers = static_cast<int>(parameters.enableTransparentLayers);
            pushConst.blurDepth = parameters.blurDepth;
        vkCmdPushConstants(commandBuffers[frameNumber], combiner.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(LayersCombinerPushConst), &pushConst);

        vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.pipeline);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.pipelineLayout, 0, 1, &combiner.descriptorSets[frameNumber], 0, nullptr);
        vkCmdDraw(commandBuffers[frameNumber], 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

}
