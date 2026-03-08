#include "bloom.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::workflows {

struct BloomPushConst{
    alignas(4) float srcWidth;
    alignas(4) float srcHeight;
    alignas(4) float filterRadius;
    alignas(4) float strength;
};

BloomGraphics::BloomGraphics(BloomParameters& parameters)
    : parameters(parameters), downsample(parameters), upsample(parameters)
{}

void BloomGraphics::Downsample::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(utils::vkDefault::imageComputeLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutBinding storageBinding{};
        storageBinding.binding = static_cast<uint32_t>(bindings.size());
        storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        storageBinding.descriptorCount = 1;
        storageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        storageBinding.pImmutableSamplers = VK_NULL_HANDLE;
    bindings.push_back(storageBinding);

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto compShader = utils::vkDefault::ComputeShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Compute));

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(BloomPushConst);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

    VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = compShader;
        pipelineInfo.layout = pipelineLayout;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    uint32_t setCount = (parameters.attachmentsCount - 1) * parameters.imageInfo.Count;
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, setCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, setCount);
}

void BloomGraphics::Upsample::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(utils::vkDefault::imageComputeLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    VkDescriptorSetLayoutBinding storageBinding{};
        storageBinding.binding = static_cast<uint32_t>(bindings.size());
        storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        storageBinding.descriptorCount = 1;
        storageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        storageBinding.pImmutableSamplers = VK_NULL_HANDLE;
    bindings.push_back(storageBinding);

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto compShader = utils::vkDefault::ComputeShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Compute));

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(BloomPushConst);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

    VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = compShader;
        pipelineInfo.layout = pipelineLayout;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    uint32_t setCount = (parameters.attachmentsCount - 1) * parameters.imageInfo.Count;
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, setCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, setCount);
}

void BloomGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        frames.resize(parameters.attachmentsCount);

        VkSamplerCreateInfo bloomSampler = utils::vkDefault::sampler();
        bloomSampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        bloomSampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        bloomSampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        uint32_t w = parameters.imageInfo.Extent.width;
        uint32_t h = parameters.imageInfo.Extent.height;
        for (uint32_t i = 0; i < parameters.attachmentsCount; i++) {
            const utils::vkDefault::ImageInfo mipInfo = {
                parameters.imageInfo.Count,
                VK_FORMAT_R32G32B32A32_SFLOAT,
                { std::max(w, 1u), std::max(h, 1u) },
                parameters.imageInfo.Samples
            };
            frames[i] = utils::Attachments(physicalDevice, device, mipInfo,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                {{0.0f, 0.0f, 0.0f, 0.0f}}, bloomSampler);
            w /= 2;
            h /= 2;
        }

        aDatabase.addAttachmentData(parameters.out.bloom, parameters.enable, &frames[0]);

        const workflows::ShaderNames downsampleShaderNames = {
            {workflows::ShaderType::Compute, "bloom/bloomDownsampleComp.spv"}
        };
        downsample.create(downsampleShaderNames, device, VK_NULL_HANDLE);

        const workflows::ShaderNames upsampleShaderNames = {
            {workflows::ShaderType::Compute, "bloom/bloomUpsampleComp.spv"}
        };
        upsample.create(upsampleShaderNames, device, VK_NULL_HANDLE);

        created = true;
    }
}

void BloomGraphics::updateDescriptors(const utils::BuffersDatabase&, const utils::AttachmentsDatabase& aDatabase)
{
    if(!parameters.enable || !created) return;

    srcAttachment = aDatabase.get(parameters.in.bloom);

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++) {
        for (uint32_t level = 0; level < parameters.attachmentsCount - 1; level++) {
            uint32_t setIndex = level * parameters.imageInfo.Count + i;

            // Downsample: read frames[level], write frames[level+1]
            {
                auto srcInfo = frames[level].descriptorImageInfo(i);
                srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo dstInfo{};
                    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    dstInfo.imageView = frames[level + 1].imageView(i);
                    dstInfo.sampler = VK_NULL_HANDLE;

                utils::descriptorSet::Writes writes;
                utils::descriptorSet::write(writes, downsample.descriptorSets[setIndex], srcInfo);
                utils::descriptorSet::write(writes, downsample.descriptorSets[setIndex], dstInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
                utils::descriptorSet::update(device, writes);
            }

            // Upsample: read frames[level+1], write frames[level]
            {
                auto srcInfo = frames[level + 1].descriptorImageInfo(i);
                srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo dstInfo{};
                    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    dstInfo.imageView = frames[level].imageView(i);
                    dstInfo.sampler = VK_NULL_HANDLE;

                utils::descriptorSet::Writes writes;
                utils::descriptorSet::write(writes, upsample.descriptorSets[setIndex], srcInfo);
                utils::descriptorSet::write(writes, upsample.descriptorSets[setIndex], dstInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
                utils::descriptorSet::update(device, writes);
            }
        }
    }
}

void BloomGraphics::updateCommandBuffer(uint32_t frameNumber){
    if(!parameters.enable || !created) return;

    const uint32_t mipCount = parameters.attachmentsCount;

    // Copy src into frames[0]
    utils::texture::transitionLayout(commandBuffers[frameNumber], srcAttachment->image(frameNumber),
        parameters.inputImageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    utils::texture::transitionLayout(commandBuffers[frameNumber], frames[0].image(frameNumber),
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);

    utils::texture::copy(commandBuffers[frameNumber], srcAttachment->image(frameNumber), frames[0].image(frameNumber),
        { parameters.imageInfo.Extent.width, parameters.imageInfo.Extent.height, 1 }, 1);

    utils::texture::transitionLayout(commandBuffers[frameNumber], srcAttachment->image(frameNumber),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, parameters.inputImageLayout, VK_REMAINING_MIP_LEVELS, 0, 1);

    // Transition frames[0] to GENERAL for compute, rest to GENERAL
    utils::texture::transitionLayout(commandBuffers[frameNumber], frames[0].image(frameNumber),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    for (uint32_t i = 1; i < mipCount; i++) {
        utils::texture::transitionLayout(commandBuffers[frameNumber], frames[i].image(frameNumber),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_REMAINING_MIP_LEVELS, 0, 1);
    }

    // Downsample pass: frames[0] -> frames[1] -> ... -> frames[N-1]
    uint32_t w = parameters.imageInfo.Extent.width;
    uint32_t h = parameters.imageInfo.Extent.height;

    vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, downsample.pipeline);
    for (uint32_t level = 0; level < mipCount - 1; level++) {
        uint32_t setIndex = level * parameters.imageInfo.Count + frameNumber;

        BloomPushConst pushConst{ static_cast<float>(w), static_cast<float>(h), parameters.filterRadius, parameters.strength };
        vkCmdPushConstants(commandBuffers[frameNumber], downsample.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomPushConst), &pushConst);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, downsample.pipelineLayout, 0, 1, &downsample.descriptorSets[setIndex], 0, nullptr);

        uint32_t dstW = std::max(w / 2, 1u);
        uint32_t dstH = std::max(h / 2, 1u);
        vkCmdDispatch(commandBuffers[frameNumber], (dstW + 7) / 8, (dstH + 7) / 8, 1);

        VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffers[frameNumber],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);

        w = dstW;
        h = dstH;
    }

    // Upsample pass: frames[N-1] -> frames[N-2] -> ... -> frames[0] (additive)
    vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, upsample.pipeline);
    for (int32_t level = static_cast<int32_t>(mipCount) - 2; level >= 0; level--) {
        uint32_t setIndex = static_cast<uint32_t>(level) * parameters.imageInfo.Count + frameNumber;

        uint32_t dstW = std::max(parameters.imageInfo.Extent.width >> level, 1u);
        uint32_t dstH = std::max(parameters.imageInfo.Extent.height >> level, 1u);

        BloomPushConst pushConst{ 0.0f, 0.0f, parameters.filterRadius, parameters.strength };
        vkCmdPushConstants(commandBuffers[frameNumber], upsample.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomPushConst), &pushConst);
        vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, upsample.pipelineLayout, 0, 1, &upsample.descriptorSets[setIndex], 0, nullptr);

        vkCmdDispatch(commandBuffers[frameNumber], (dstW + 7) / 8, (dstH + 7) / 8, 1);

        VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffers[frameNumber],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    // Transition frames[0] to SHADER_READ_ONLY for downstream consumers
    utils::texture::transitionLayout(commandBuffers[frameNumber], frames[0].image(frameNumber),
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_REMAINING_MIP_LEVELS, 0, 1);
}

} // moon::workflows
