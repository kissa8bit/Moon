#include "layersCombiner.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

#include <iostream>

namespace moon::deferredGraphics {

LayersCombiner::LayersCombiner(LayersCombinerParameters& parameters) : combiner(parameters), parameters(parameters) {}

void LayersCombiner::createAttachments(utils::AttachmentsDatabase& aDatabase)
{
    const utils::vkDefault::ImageInfo f32Info = { parameters.imageInfo.Count, VK_FORMAT_R32G32B32A32_SFLOAT, parameters.imageInfo.Extent, VK_SAMPLE_COUNT_1_BIT };
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    frame.color = utils::Attachments(physicalDevice, device, f32Info, usage);
    frame.bloom = utils::Attachments(physicalDevice, device, f32Info, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    frame.blur = utils::Attachments(physicalDevice, device, f32Info, usage);
    aDatabase.addAttachmentData(parameters.out.color, parameters.enable, &frame.color);
    aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &frame.bloom);
    aDatabase.addAttachmentData(parameters.out.blur, parameters.enable, &frame.blur);
}

void LayersCombiner::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(frame.color.format(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        utils::Attachments::imageDescription(frame.bloom.format(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        utils::Attachments::imageDescription(frame.blur.format(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    };

    utils::vkDefault::SubpassInfos subpassInfos = utils::vkDefault::subpassInfos(attachments.size());

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
    dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies.back().dstSubpass = 0;
    dependencies.back().srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies.back().srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies.back().dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
                                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies.back().dstAccessMask = VK_ACCESS_SHADER_READ_BIT|
                                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

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
	const uint32_t layersCount = parameters.layersCount.get();
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::bufferFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), layersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), layersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), layersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), layersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), layersCount));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const uint32_t specializationData = layersCount;
    std::vector<VkSpecializationMapEntry> specializationMapEntry;
        specializationMapEntry.push_back(VkSpecializationMapEntry{});
        specializationMapEntry.back().constantID = static_cast<uint32_t>(specializationMapEntry.size() - 1);
        specializationMapEntry.back().offset = 0;
        specializationMapEntry.back().size = sizeof(uint32_t);
    VkSpecializationInfo specializationInfo{};
        specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntry.size());
        specializationInfo.pMapEntries = specializationMapEntry.data();
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
        std::vector<VkDescriptorImageInfo> colorLayersImageInfos(parameters.layersCount.get());
        std::vector<VkDescriptorImageInfo> bloomLayersImageInfos(parameters.layersCount.get());
        std::vector<VkDescriptorImageInfo> positionLayersImageInfos(parameters.layersCount.get());
        std::vector<VkDescriptorImageInfo> normalLayersImageInfos(parameters.layersCount.get());
        std::vector<VkDescriptorImageInfo> depthLayersImageInfos(parameters.layersCount.get());

        for (LayerIndex index{ 0 }; index < parameters.layersCount; index++) {
            auto pref = layerPrefix(index);

            colorLayersImageInfos.at(index.get()) = aDatabase.descriptorImageInfo(pref + parameters.in.color, i);
            bloomLayersImageInfos.at(index.get()) = aDatabase.descriptorImageInfo(pref + parameters.in.bloom, i);
            positionLayersImageInfos.at(index.get()) = aDatabase.descriptorImageInfo(pref + parameters.in.position, i);
            normalLayersImageInfos.at(index.get()) = aDatabase.descriptorImageInfo(pref + parameters.in.normal, i);
            depthLayersImageInfos.at(index.get()) = aDatabase.descriptorImageInfo(pref + parameters.in.depth, i, parameters.in.defaultDepthTexture);
        }

        auto descriptorSet = combiner.descriptorSets.at(i);

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSet, bDatabase.descriptorBufferInfo(parameters.in.camera, i));
        utils::descriptorSet::write(writes, descriptorSet, colorLayersImageInfos);
        utils::descriptorSet::write(writes, descriptorSet, bloomLayersImageInfos);
        utils::descriptorSet::write(writes, descriptorSet, positionLayersImageInfos);
        utils::descriptorSet::write(writes, descriptorSet, normalLayersImageInfos);
        utils::descriptorSet::write(writes, descriptorSet, depthLayersImageInfos);
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.skyboxColor, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.skyboxBloom, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.scattering, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.sslr, i));
        utils::descriptorSet::update(device, writes);
    }
}

void LayersCombiner::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    std::vector<VkClearValue> clearValues = { frame.color.clearValue(), frame.bloom.clearValue(), frame.blur.clearValue() };

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers.at(frameNumber);
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers.at(frameNumber), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers.at(frameNumber), VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.pipeline);
        vkCmdBindDescriptorSets(commandBuffers.at(frameNumber), VK_PIPELINE_BIND_POINT_GRAPHICS, combiner.pipelineLayout, 0, 1, &combiner.descriptorSets.at(frameNumber), 0, nullptr);
        vkCmdDraw(commandBuffers.at(frameNumber), 6, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffers.at(frameNumber));
}

} // moon::deferredGraphics
