#include "boundingBox.h"

#include <utils/vkdefault.h>
#include <utils/operations.h>

#include <implementations/BaseObject.h>

namespace moon::workflows {

BoundingBoxGraphics::BoundingBoxGraphics(BoundingBoxParameters& parameters, const interfaces::Objects* objects) : parameters(parameters), box(parameters, objects) {}

void BoundingBoxGraphics::createAttachments(utils::AttachmentsDatabase& aDatabase){
    const utils::vkDefault::ImageInfo info = { parameters.imageInfo.Count, VK_FORMAT_R8G8B8A8_UNORM, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    utils::createAttachments(physicalDevice, device, info, 1, &frame);
    aDatabase.addAttachmentData(parameters.out.boundingBox, parameters.enable, &frame);
}

void BoundingBoxGraphics::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(VK_FORMAT_R8G8B8A8_UNORM)
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

void BoundingBoxGraphics::createFramebuffers(){
    framebuffers.resize(parameters.imageInfo.Count);
    for(size_t i = 0; i < parameters.imageInfo.Count; i++){
        std::vector<VkImageView> pAttachments = {frame.imageView(i)};
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(pAttachments.size());
            framebufferInfo.pAttachments = pAttachments.data();
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void BoundingBoxGraphics::BoundingBox::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
    objectDescriptorSetLayout = implementations::BaseObject::createDescriptorSetLayout(device);
    skeletonDescriptorSetLayout = interfaces::Skeleton::descriptorSetLayout(device);

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment));
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = utils::vkDefault::viewport({0,0}, parameters.imageInfo.Extent);
    VkRect2D scissor = utils::vkDefault::scissor({0,0}, parameters.imageInfo.Extent);
    VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilDisable();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = utils::vkDefault::vertexInputState();

    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_SHADER_STAGE_ALL;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(math::box);
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout,
        objectDescriptorSetLayout,
        skeletonDescriptorSetLayout
    };
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
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void BoundingBoxGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "boundingBox/boundingBoxVert.spv"},
            {workflows::ShaderType::Fragment, "boundingBox/boundingBoxFrag.spv"}
        };
        box.create(shaderNames, device, renderPass);
        created = true;
    }
}

void BoundingBoxGraphics::updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase&) {
    if (!parameters.enable || !created) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++) {
        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, box.descriptorSets[i], bDatabase.descriptorBufferInfo(parameters.in.camera, i));
        utils::descriptorSet::update(device, writes);
    }
}

void BoundingBoxGraphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

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

        box.render(frameNumber, commandBuffers[frameNumber]);

    vkCmdEndRenderPass(commandBuffers[frameNumber]);
}

void BoundingBoxGraphics::BoundingBox::render(uint32_t frameNumber, VkCommandBuffer commandBuffers){
    if (!objects) return;

    vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for(const auto& object: *objects){
        if (!object) continue;

        const auto mask = object->objectMask();
        const auto property = mask.property();
        const auto model = object->model();

        if (!property.has(interfaces::ObjectProperty::enable)) continue;
        if (!model) continue;

        utils::vkDefault::DescriptorSets descriptors = { descriptorSets[frameNumber], object->getDescriptorSet(frameNumber) };
        model->renderBB(object->getInstanceNumber(frameNumber), commandBuffers, pipelineLayout, descriptors);
    }
}

}
