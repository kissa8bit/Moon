#include "selector.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>

namespace moon::workflows {

SelectorGraphics::SelectorGraphics(SelectorParameters& parameters, utils::Cursor** cursor) : parameters(parameters), selector(parameters), cursor(cursor) {}

void SelectorGraphics::Selector::create(const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::bufferComputeLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageComputeLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::imageComputeLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);

    const auto compShader = utils::vkDefault::ComputeShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Compute));

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { descriptorSetLayout };
    pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts);

    VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = compShader;
        pipelineInfo.layout = pipelineLayout;
    pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void SelectorGraphics::create(const utils::vkDefault::CommandPool& commandPool, utils::AttachmentsDatabase& aDatabase) {
    commandBuffers = commandPool.allocateCommandBuffers(parameters.imageInfo.Count);
    if(parameters.enable && !created){
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Compute, "selector/selectorComp.spv"}
        };
        selector.create(shaderNames, device, VK_NULL_HANDLE);
        created = true;
    }
}

void SelectorGraphics::updateDescriptors(const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase) {
    if (!parameters.enable || !created || !cursor) return;

    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++) {
        auto descriptorSet = selector.descriptorSets.at(i);

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR_T(writes, descriptorSet, (*cursor)->descriptorBufferInfo(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.normal, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.in.depth, i, parameters.in.defaultDepthTexture));
        utils::descriptorSet::update(device, writes);
    }
}

void SelectorGraphics::updateCommandBuffer(uint32_t frameNumber){
    if (!parameters.enable || !created) return;

    vkCmdBindPipeline(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, selector.pipeline);
    vkCmdBindDescriptorSets(commandBuffers[frameNumber], VK_PIPELINE_BIND_POINT_COMPUTE, selector.pipelineLayout, 0, 1, &selector.descriptorSets[frameNumber], 0, nullptr);
    vkCmdDispatch(commandBuffers[frameNumber], 1, 1, 1);
}

} // moon::workflows
