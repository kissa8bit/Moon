#include "boundingBoxGraphics.h"

#include <cstring>

#include <utils/operations.h>
#include <utils/vkdefault.h>

#include <math/linearAlgebra.h>

namespace moon::rayTracingGraphics {

struct CameraBuffer{
    alignas(16) math::mat4 proj;
    alignas(16) math::mat4 view;
};

void BoundingBoxGraphics::createAttachments() {
    moon::utils::createAttachments(physicalDevice, device, image, 1, &frame);
}

void BoundingBoxGraphics::BoundingBoxGraphics::createRenderPass(){
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        moon::utils::Attachments::imageDescription(image.Format)
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
    framebuffers.resize(image.Count);
    for(size_t i = 0; i < image.Count; i++){
        std::vector<VkImageView> pAttachments = {frame.imageView(i)};
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(pAttachments.size());
            framebufferInfo.pAttachments = pAttachments.data();
            framebufferInfo.width = image.Extent.width;
            framebufferInfo.height = image.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers[i] = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void BoundingBoxGraphics::createDescriptorSetLayout(){
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(moon::utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
}

void BoundingBoxGraphics::createPipeline(){
    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, vertShaderPath);
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, fragShaderPath);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = { vertShader, fragShader };

    VkViewport viewport = moon::utils::vkDefault::viewport({0,0}, image.Extent);
    VkRect2D scissor = moon::utils::vkDefault::scissor({0,0}, image.Extent);
    VkPipelineViewportStateCreateInfo viewportState = moon::utils::vkDefault::viewportState(&viewport, &scissor);
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = moon::utils::vkDefault::inputAssembly();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = moon::utils::vkDefault::vertexInputState();
    VkPipelineRasterizationStateCreateInfo rasterizer = moon::utils::vkDefault::rasterizationState();
    VkPipelineMultisampleStateCreateInfo multisampling = moon::utils::vkDefault::multisampleState();
    VkPipelineDepthStencilStateCreateInfo depthStencil = moon::utils::vkDefault::depthStencilDisable();

    rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        moon::utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    VkPipelineColorBlendStateCreateInfo colorBlending = moon::utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()),colorBlendAttachment.data());

    std::vector<VkPushConstantRange> pushConstantRange;
        pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(cuda::rayTracing::cbox);
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
    pipelineInfo.back().pDepthStencilState = &depthStencil;
    pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);
}


void BoundingBoxGraphics::createDescriptors(){
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, image.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, image.Count);

    for (uint32_t i = 0; i < image.Count; i++)
    {
        const auto bufferInfo = cameraBuffers[i].descriptorBufferInfo();

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, descriptorSets[i], bufferInfo);
        utils::descriptorSet::update(device, writes);
    }
}

void BoundingBoxGraphics::create(VkPhysicalDevice physicalDevice, VkDevice device, const moon::utils::vkDefault::ImageInfo& image, const std::filesystem::path& shadersPath){
    if(!enable) return;

    this->physicalDevice = physicalDevice;
    this->device = device;
    this->image = image;
    vertShaderPath = shadersPath / "boundingBox/boundingBoxVert.spv";
    fragShaderPath = shadersPath / "boundingBox/boundingBoxFrag.spv";

    cameraBuffers.resize(image.Count);
    for(auto& buffer: cameraBuffers) {
        buffer = utils::vkDefault::Buffer(physicalDevice, device, sizeof(CameraBuffer), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    createAttachments();
    createRenderPass();
    createFramebuffers();
    createDescriptorSetLayout();
    createPipeline();
    createDescriptors();
}

void BoundingBoxGraphics::update(uint32_t imageIndex){
    if(!enable) return;

    cuda::rayTracing::Camera hostCam = cuda::rayTracing::to_host(*camera);
    const float fov = 2.0f * std::atan(hostCam.matrixScale / hostCam.matrixOffset);
    const auto& u =  normal(hostCam.horizontal);
    const auto& v =  normal(hostCam.vertical);
    const auto& n = -normal(hostCam.viewRay.getDirection());
    const auto& c = hostCam.viewRay.getOrigin();

    math::mat4 projMatrix = moon::math::perspective(fov, hostCam.aspect, hostCam.matrixOffset);
    math::mat4 viewMatrix(
        math::vec4(u[0], u[1], u[2], - (c[0]*u[0] + c[1]*u[1] + c[2]*u[2])),
        math::vec4(v[0], v[1], v[2], - (c[0]*v[0] + c[1]*v[1] + c[2]*v[2])),
        math::vec4(n[0], n[1], n[2], - (c[0]*n[0] + c[1]*n[1] + c[2]*n[2])),
        math::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    CameraBuffer buffer{transpose(projMatrix), transpose(viewMatrix)};
    cameraBuffers[imageIndex].copy(&buffer);
}

void BoundingBoxGraphics::render(VkCommandBuffer commandBuffer, uint32_t imageIndex) const {
    if(!enable) return;

    std::vector<VkClearValue> clearValues = {frame.clearValue()};

    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = image.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, NULL);
    for(const auto& box: boxes){
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(cuda::rayTracing::cbox), &box);
        vkCmdDraw(commandBuffer, 24, 1, 0, 0);
    }

    vkCmdEndRenderPass(commandBuffer);
}

const moon::utils::Attachments& BoundingBoxGraphics::getAttachments() const {
    return frame;
}

void BoundingBoxGraphics::bind(const cuda::rayTracing::cbox& box){
    boxes.push_back(box);
}

void BoundingBoxGraphics::clear(){
    boxes.clear();
}

void BoundingBoxGraphics::bind(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* camera){
    this->camera = camera;
}

} // moon::rayTracingGraphics
