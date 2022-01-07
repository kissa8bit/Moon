#include "graphics.h"

#include <utils/vkdefault.h>
#include <utils/depthMap.h>
#include <utils/operations.h>

#include "deferredAttachments.h"

namespace moon::deferredGraphics {

Graphics::Lighting::Lighting(const GraphicsParameters& parameters, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps)
    : parameters(parameters), lightSources(lightSources), depthMaps(depthMaps)
{}

void Graphics::Lighting::create(VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
    shadowDescriptorSetLayout = utils::DepthMap::createDescriptorSetLayout(device);

    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingCircleVert.spv"},
            {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingCircleFrag.spv"}
        };
        createPipeline(interfaces::LightType::spotCircle, shaderNames, device, renderPass);
    }
    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingSquareVert.spv"},
            {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingSquareFrag.spv"}
        };
        createPipeline(interfaces::LightType::spotSquare, shaderNames, device, renderPass);
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void Graphics::Lighting::update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++){
        auto descriptorSet = descriptorSets[i];

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.out.position, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.out.normal, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.out.color, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(parameters.out.depth, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR(writes, descriptorSet, bDatabase.descriptorBufferInfo(parameters.in.camera, i));
        utils::descriptorSet::update(device, writes);
    }
}

void Graphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const
{
    for(const auto& lightSource: *lightSources){
        if (!lightSource) continue;

        const auto& depthMap = depthMaps->at(lightSource);

        const auto mask = lightSource->lightMask();
        const auto type = mask.type();

        const auto& pipelineDesc = pipelineDescs.at(type);

        const utils::vkDefault::DescriptorSets descriptors = { descriptorSets.at(frameNumber), depthMap.descriptorSets().at(frameNumber) };
        lightSource->render(frameNumber, commandBuffer, descriptors, pipelineDesc.pipelineLayout, pipelineDesc.pipeline);
    }
}

}
