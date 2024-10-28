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
    lightDescriptorSetLayoutMap[interfaces::Light::Type::spot] = interfaces::Light::createDescriptorSetLayout(device);
    shadowDescriptorSetLayout = utils::DepthMap::createDescriptorSetLayout(device);

    const workflows::ShaderNames shaderNames = {
        {workflows::ShaderType::Vertex, "spotLightingPass/spotLightingVert.spv"},
        {workflows::ShaderType::Fragment, "spotLightingPass/spotLightingFrag.spv"}
    };
    createPipeline(interfaces::Light::Type::spot, shaderNames, device, renderPass);

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void Graphics::Lighting::update(
    VkDevice device,
    const utils::BuffersDatabase& bDatabase,
    const utils::AttachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++){
        std::vector<VkDescriptorImageInfo> imageInfos;
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.position, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.normal, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.color, i));
        imageInfos.push_back(aDatabase.descriptorImageInfo(parameters.out.depth, i));

        VkDescriptorBufferInfo bufferInfo = bDatabase.descriptorBufferInfo(parameters.in.camera, i);

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(auto& imageInfo: imageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
                descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites.back().dstSet = descriptorSets[i];
                descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
                descriptorWrites.back().dstArrayElement = 0;
                descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
                descriptorWrites.back().descriptorCount = 1;
                descriptorWrites.back().pImageInfo = &imageInfo;
        }
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptorSets[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Graphics::Lighting::render(uint32_t frameNumber, VkCommandBuffer commandBuffer) const
{
    for(auto& lightSource: *lightSources){
        const auto& depthMap = depthMaps->at(lightSource);
        uint8_t mask = lightSource->pipelineFlagBits();
        lightSource->render(
            frameNumber,
            commandBuffer,
            {descriptorSets[frameNumber], depthMap.descriptorSets()[frameNumber]},
            pipelineLayoutMap.at(mask),
            pipelineMap.at(mask));
    }
}

}
