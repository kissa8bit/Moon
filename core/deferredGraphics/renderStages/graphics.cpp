#include "graphics.h"

#include <utils/operations.h>

namespace moon::deferredGraphics {

Graphics::Graphics(
    GraphicsParameters& parameters,
    LayerIndex layerIndex,
    const interfaces::Objects* object,
    const interfaces::Lights* lightSources,
    const interfaces::DepthMaps* depthMaps)
    :   parameters(parameters),
        layerIndex(layerIndex),
        base(parameters, layerIndex, object),
        outlining(base),
        lighting(parameters, layerIndex, lightSources, depthMaps),
        ambientLighting(lighting)
{}

void Graphics::createAttachments(utils::AttachmentsDatabase& aDatabase) {
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    const utils::vkDefault::ImageInfo u8Info = { parameters.imageInfo.Count, VK_FORMAT_R8G8B8A8_UNORM, parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    const utils::vkDefault::ImageInfo f32Info = { parameters.imageInfo.Count, VK_FORMAT_R32G32B32A32_SFLOAT, parameters.imageInfo.Extent, parameters.imageInfo.Samples };

    deferredAttachments.image() = utils::Attachments(physicalDevice, device, f32Info, usage);
    deferredAttachments.bloom() = utils::Attachments(physicalDevice, device, f32Info, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    deferredAttachments.position() = utils::Attachments(physicalDevice, device, f32Info, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    deferredAttachments.normal() = utils::Attachments(physicalDevice, device, f32Info, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    deferredAttachments.color() = utils::Attachments(physicalDevice, device, u8Info, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
    deferredAttachments.emission() = utils::Attachments(physicalDevice, device, u8Info, usage | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);

    const utils::vkDefault::ImageInfo depthImage = { parameters.imageInfo.Count, utils::image::depthStencilFormat(physicalDevice), parameters.imageInfo.Extent, parameters.imageInfo.Samples };
    deferredAttachments.depth() = utils::Attachments(physicalDevice, device, depthImage, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, { { 1.0f, 0 } }, utils::vkDefault::depthSampler());

    const auto pref = layerPrefix(layerIndex);
    aDatabase.addAttachmentData(pref + parameters.out.image, parameters.enable, &deferredAttachments.image());
    aDatabase.addAttachmentData(pref + parameters.out.bloom, parameters.enable, &deferredAttachments.bloom());
    aDatabase.addAttachmentData(pref + parameters.out.position, parameters.enable, &deferredAttachments.position());
    aDatabase.addAttachmentData(pref + parameters.out.normal, parameters.enable, &deferredAttachments.normal());
    aDatabase.addAttachmentData(pref + parameters.out.color, parameters.enable, &deferredAttachments.color());
    aDatabase.addAttachmentData(pref + parameters.out.emission, parameters.enable, &deferredAttachments.emission());
    aDatabase.addAttachmentData(pref + parameters.out.depth, parameters.enable, &deferredAttachments.depth());
}

void Graphics::createRenderPass()
{
    utils::vkDefault::RenderPass::AttachmentDescriptions attachments = {
        utils::Attachments::imageDescription(deferredAttachments.image().format()),
        utils::Attachments::imageDescription(deferredAttachments.bloom().format()),
        utils::Attachments::imageDescription(deferredAttachments.position().format()),
        utils::Attachments::imageDescription(deferredAttachments.normal().format()),
        utils::Attachments::imageDescription(deferredAttachments.color().format()),
        utils::Attachments::imageDescription(deferredAttachments.emission().format()),
        utils::Attachments::depthStencilDescription(deferredAttachments.depth().format())
    };

    utils::vkDefault::SubpassInfos subpassInfos;

    auto& geometry = subpassInfos.emplace_back();
    geometry.out = {
        VkAttachmentReference{DeferredAttachments::positionIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::normalIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::colorIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::emissionIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };
    geometry.depth = {
        VkAttachmentReference{DeferredAttachments::depthIndex(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}
    };

    auto& lighting = subpassInfos.emplace_back();
    lighting.out = {
        VkAttachmentReference{DeferredAttachments::imageIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::bloomIndex(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };
    lighting.in = {
        VkAttachmentReference{DeferredAttachments::positionIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::normalIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::colorIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::emissionIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkAttachmentReference{DeferredAttachments::depthIndex(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
    };

    utils::vkDefault::RenderPass::SubpassDependencies dependencies;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies.back().dstSubpass = 0;
        dependencies.back().srcStageMask =  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT| VK_PIPELINE_STAGE_HOST_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        dependencies.back().dstStageMask =  VK_PIPELINE_STAGE_VERTEX_INPUT_BIT|
                                            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT|
                                            VK_ACCESS_INDEX_READ_BIT|
                                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies.push_back(VkSubpassDependency{});
        dependencies.back().srcSubpass = 0;
        dependencies.back().dstSubpass = 1;
        dependencies.back().srcStageMask =  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|
                                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies.back().srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies.back().dstStageMask =  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies.back().dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT|
                                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
                                            VK_ACCESS_UNIFORM_READ_BIT;

    renderPass = utils::vkDefault::RenderPass(device, attachments, subpassInfos, dependencies);
}

void Graphics::createFramebuffers() {
    framebuffers.resize(parameters.imageInfo.Count);
    for (size_t imageIndex = 0; imageIndex < parameters.imageInfo.Count; imageIndex++){
        auto views = deferredAttachments.views(imageIndex);
        VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(views.size());
            framebufferInfo.pAttachments = views.data();
            framebufferInfo.width = parameters.imageInfo.Extent.width;
            framebufferInfo.height = parameters.imageInfo.Extent.height;
            framebufferInfo.layers = 1;
        framebuffers.at(imageIndex) = utils::vkDefault::Framebuffer(device, framebufferInfo);
    }
}

void Graphics::createPipelines() {
    const workflows::ShaderNames baseShaderNames = {
        {workflows::ShaderType::Vertex, "base/baseVert.spv"},
        {workflows::ShaderType::Fragment, "base/baseFrag.spv"}
    };
    base.create(interfaces::ObjectType::base, baseShaderNames, device, renderPass);

    const workflows::ShaderNames baseSimpleShaderNames = {
        {workflows::ShaderType::Vertex, "base/baseSimpleVert.spv"},
        {workflows::ShaderType::Fragment, "base/baseFrag.spv"}
    };
    base.create(interfaces::ObjectType::baseSimple, baseSimpleShaderNames, device, renderPass);

    const workflows::ShaderNames outliningShaderNames = {
        {workflows::ShaderType::Vertex, "outlining/outliningVert.spv"},
        {workflows::ShaderType::Fragment, "outlining/outliningFrag.spv"}
    };
    outlining.create(interfaces::ObjectType::base, outliningShaderNames, device, renderPass);

    const workflows::ShaderNames outliningSimpleShaderNames = {
        {workflows::ShaderType::Vertex, "outlining/outliningSimpleVert.spv"},
        {workflows::ShaderType::Fragment, "outlining/outliningFrag.spv"}
    };
    outlining.create(interfaces::ObjectType::baseSimple, outliningSimpleShaderNames, device, renderPass);

    lighting.create(device, renderPass);

    const workflows::ShaderNames ambientLightingShaderNames = {
        {workflows::ShaderType::Vertex, "ambientLightingPass/ambientLightingVert.spv"},
        {workflows::ShaderType::Fragment, "ambientLightingPass/ambientLightingFrag.spv"}
    };
    ambientLighting.create(ambientLightingShaderNames, device, renderPass);
}

void Graphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        createAttachments(aDatabase);
        createRenderPass();
        createFramebuffers();
        createPipelines();
        created = true;
    }
}

void Graphics::updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) {
    if (!parameters.enable || !created) return;
    base.update(device, bDatabase, aDatabase);
    lighting.update(device, bDatabase, aDatabase);
}

void Graphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    const std::vector<VkClearValue> clearValues = deferredAttachments.clearValues();
    VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers.at(frameNumber);
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = parameters.imageInfo.Extent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffers.at(frameNumber), &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        base.render(frameNumber,commandBuffers.at(frameNumber));
        outlining.render(frameNumber,commandBuffers.at(frameNumber));

    vkCmdNextSubpass(commandBuffers.at(frameNumber), VK_SUBPASS_CONTENTS_INLINE);

        lighting.render(frameNumber,commandBuffers.at(frameNumber));
        ambientLighting.render(frameNumber,commandBuffers.at(frameNumber));

    vkCmdEndRenderPass(commandBuffers.at(frameNumber));
}

} // moon::deferredGraphics
