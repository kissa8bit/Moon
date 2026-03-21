#include "graphics.h"

#include <utils/vkdefault.h>
#include <utils/depthMap.h>
#include <utils/operations.h>
#include <interfaces/camera.h>

#include "deferredAttachments.h"

namespace moon::deferredGraphics {

Graphics::Lighting::Lighting(const GraphicsParameters& parameters, LayerIndex layerIndex, const interfaces::Lights* lightSources, const interfaces::DepthMaps* depthMaps)
    : parameters(parameters), layerIndex(layerIndex), lightSources(lightSources), depthMaps(depthMaps)
{}

void Graphics::Lighting::create(VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
        bindings.push_back(utils::vkDefault::inAttachmentFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings);
    shadowDescriptorSetLayout = utils::DepthMap::createDescriptorSetLayout(device);

    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingVert.spv"},
            {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingCircleFrag.spv"}
        };
        createPipeline(interfaces::LightType::spotCircle, shaderNames, device, renderPass);
    }
    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingVert.spv"},
            {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingSquareFrag.spv"}
        };
        createPipeline(interfaces::LightType::spotSquare, shaderNames, device, renderPass);
    }
    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "pointLightingPass/pointLightingVert.spv"},
            {workflows::ShaderType::Fragment, "pointLightingPass/pointLightingFrag.spv"}
        };
        createPipeline(interfaces::LightType::pointLight, shaderNames, device, renderPass);
    }
    {
        const workflows::ShaderNames shaderNames = {
            {workflows::ShaderType::Vertex, "directionalLightingPass/directionalLightingVert.spv"},
            {workflows::ShaderType::Fragment, "directionalLightingPass/directionalLightingFrag.spv"}
        };
        createPipeline(interfaces::LightType::directional, shaderNames, device, renderPass);
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void Graphics::Lighting::update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++){
        auto descriptorSet = descriptorSets[i];

        auto pref = layerPrefix(layerIndex);

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(pref + parameters.out.normal, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(pref + parameters.out.color, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        WRITE_DESCRIPTOR_T(writes, descriptorSet, aDatabase.descriptorImageInfo(pref + parameters.out.emission, i), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        auto depthInfo = aDatabase.descriptorImageInfo(pref + parameters.out.depth, i);
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        WRITE_DESCRIPTOR_T(writes, descriptorSet, depthInfo, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
        utils::descriptorSet::update(device, writes);
    }
}

void Graphics::Lighting::render(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) const
{
    if (!lightSources) return;
    if (!depthMaps) return;
    if (!parameters.in.camera || !*parameters.in.camera) return;

    const auto frameNumber = resourceIndex.get();
    for(const auto& lightSource: *lightSources){
        if (!lightSource) continue;

        const auto& depthMap = depthMaps->find(lightSource) != depthMaps->end() ? depthMaps->at(lightSource) : depthMaps->at(parameters.in.nullDepthMapKey);

        const auto mask = lightSource->lightMask();
        const auto type = mask.type();

        const auto& pipelineDesc = pipelineDescs.at(type);

        const utils::vkDefault::DescriptorSets descriptors = { (*parameters.in.camera)->getDescriptorSet(resourceIndex), descriptorSets.at(frameNumber), depthMap.descriptorSets().at(frameNumber) };
        lightSource->render(resourceIndex, commandBuffer, descriptors, pipelineDesc.pipelineLayout, pipelineDesc.pipeline);
    }
}

}
