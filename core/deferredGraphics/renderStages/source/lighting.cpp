#include "graphics.h"
#include "vkdefault.h"
#include "deferredAttachments.h"
#include "depthMap.h"
#include "operations.h"

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

    const workflows::ShaderNames shaderNames = {
        {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingVert.spv"},
        {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingFrag.spv"}
    };
    createPipeline(interfaces::Light::Type::spot, shaderNames, device, renderPass);

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
    for(auto& lightSource: *lightSources){
        const auto& depthMap = depthMaps->at(lightSource);
        uint8_t mask = lightSource->pipelineFlagBits();
        const utils::vkDefault::DescriptorSets descriptors = { descriptorSets[frameNumber], depthMap.descriptorSets()[frameNumber] };
        lightSource->render(frameNumber, commandBuffer, descriptors, pipelineLayoutMap.at(mask), pipelineMap.at(mask));
    }
}

}
