#include "graphics.h"

#include <utils/operations.h>
#include <utils/vkdefault.h>
#include <utils/texture.h>

#include <interfaces/model.h>
#include <interfaces/object.h>
#include <implementations/baseObject.h>

namespace moon::deferredGraphics {

Graphics::Base::Base(const GraphicsParameters& parameters, const interfaces::Objects* objects) : parameters(parameters), objects(objects) {}

void Graphics::Base::create(interfaces::ObjectType type, const workflows::ShaderNames& shadersNames, VkDevice device, VkRenderPass renderPass) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));
    bindings.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(bindings.size()), 1));

    const std::vector<VkBool32> transparencyData = {
        static_cast<VkBool32>(parameters.enableTransparency),
        static_cast<VkBool32>(parameters.transparencyPass)
    };
    std::vector<VkSpecializationMapEntry> specializationMapEntry;
    specializationMapEntry.push_back(VkSpecializationMapEntry{});
        specializationMapEntry.back().constantID = static_cast<uint32_t>(specializationMapEntry.size() - 1);
        specializationMapEntry.back().offset = 0;
        specializationMapEntry.back().size = sizeof(VkBool32);
    specializationMapEntry.push_back(VkSpecializationMapEntry{});
        specializationMapEntry.back().constantID = static_cast<uint32_t>(specializationMapEntry.size() - 1);
        specializationMapEntry.back().offset = sizeof(VkBool32);
        specializationMapEntry.back().size = sizeof(VkBool32);
    VkSpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntry.size());
        specializationInfo.pMapEntries = specializationMapEntry.data();
        specializationInfo.dataSize = sizeof(decltype(transparencyData)::value_type) * transparencyData.size();
        specializationInfo.pData = transparencyData.data();

    const auto vertShader = utils::vkDefault::VertrxShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Vertex));
    const auto fragShader = utils::vkDefault::FragmentShaderModule(device, parameters.shadersPath / shadersNames.at(workflows::ShaderType::Fragment), specializationInfo);
    const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {vertShader, fragShader};

    const auto bindingDescription = interfaces::VertexInputBindingDescriptionFromObjectType(type);
    const auto attributeDescriptions = interfaces::AttributeDescriptionsFromObjectType(type);
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    const VkViewport viewport = utils::vkDefault::viewport({0,0}, parameters.imageInfo.Extent);
    const VkRect2D scissor = utils::vkDefault::scissor({0,0}, parameters.imageInfo.Extent);
    const VkPipelineViewportStateCreateInfo viewportState = utils::vkDefault::viewportState(&viewport, &scissor);
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly = utils::vkDefault::inputAssembly();
    const VkPipelineRasterizationStateCreateInfo rasterizer = utils::vkDefault::rasterizationState(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    const VkPipelineMultisampleStateCreateInfo multisampling = utils::vkDefault::multisampleState();
    const VkPipelineDepthStencilStateCreateInfo depthStencil = utils::vkDefault::depthStencilEnable();

    const std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachment = {
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE),
        utils::vkDefault::colorBlendAttachmentState(VK_FALSE)
    };
    const VkPipelineColorBlendStateCreateInfo colorBlending = utils::vkDefault::colorBlendState(static_cast<uint32_t>(colorBlendAttachment.size()), colorBlendAttachment.data());

    auto& pipelineDesc = pipelineDescs[type];

    std::vector<VkPushConstantRange> pushConstantRange;
    pushConstantRange.push_back(VkPushConstantRange{});
        pushConstantRange.back().stageFlags = VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM;
        pushConstantRange.back().offset = 0;
        pushConstantRange.back().size = sizeof(interfaces::Material::Buffer);
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts = {
        descriptorSetLayout = utils::vkDefault::DescriptorSetLayout(device, bindings),
        objectDescriptorSetLayout = implementations::BaseObject::createDescriptorSetLayout(device),
        skeletonDescriptorSetLayout = interfaces::Skeleton::descriptorSetLayout(device),
        materialDescriptorSetLayout = interfaces::Material::descriptorSetLayout(device)
    };
    pipelineDesc.pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

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
        pipelineInfo.back().layout = pipelineDesc.pipelineLayout;
        pipelineInfo.back().renderPass = renderPass;
        pipelineInfo.back().subpass = 0;
        pipelineInfo.back().pDepthStencilState = &depthStencil;
        pipelineInfo.back().basePipelineHandle = VK_NULL_HANDLE;
    pipelineDesc.pipeline = utils::vkDefault::Pipeline(device, pipelineInfo);

    {
        VkPipelineDepthStencilStateCreateInfo outliningDepthStencil = utils::vkDefault::depthStencilEnable();
            outliningDepthStencil.stencilTestEnable = VK_TRUE;
            outliningDepthStencil.back.compareOp = VK_COMPARE_OP_ALWAYS;
            outliningDepthStencil.back.failOp = VK_STENCIL_OP_REPLACE;
            outliningDepthStencil.back.depthFailOp = VK_STENCIL_OP_REPLACE;
            outliningDepthStencil.back.passOp = VK_STENCIL_OP_REPLACE;
            outliningDepthStencil.back.compareMask = 0xff;
            outliningDepthStencil.back.writeMask = 0xff;
            outliningDepthStencil.back.reference = 1;
            outliningDepthStencil.front = outliningDepthStencil.back;

        interfaces::ObjectType outliningMask{type | interfaces::ObjectType::outlining};
        auto& pipelineDesc = pipelineDescs[outliningMask];
        pipelineDesc.pipelineLayout = utils::vkDefault::PipelineLayout(device, descriptorSetLayouts, pushConstantRange);

        std::vector<VkGraphicsPipelineCreateInfo> outliningpipelineInfo = pipelineInfo;
        outliningpipelineInfo.back().layout = pipelineDesc.pipelineLayout;
        outliningpipelineInfo.back().pDepthStencilState = &outliningDepthStencil;

        pipelineDesc.pipeline = utils::vkDefault::Pipeline(device, outliningpipelineInfo);
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device, { &descriptorSetLayout }, parameters.imageInfo.Count);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, parameters.imageInfo.Count);
}

void Graphics::Base::update(VkDevice device, const utils::BuffersDatabase& bDatabase, const utils::AttachmentsDatabase& aDatabase)
{
    for (uint32_t i = 0; i < parameters.imageInfo.Count; i++) {
        auto descriptorSet = descriptorSets.at(i);

        std::string depthId = !parameters.transparencyPass || parameters.transparencyNumber == 0 ? "" :
            (parameters.out.transparency + std::to_string(parameters.transparencyNumber - 1) + ".") + parameters.out.depth;

        utils::descriptorSet::Writes writes;
        WRITE_DESCRIPTOR(writes, descriptorSet, bDatabase.descriptorBufferInfo(parameters.in.camera, i));
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorEmptyInfo());
        WRITE_DESCRIPTOR(writes, descriptorSet, aDatabase.descriptorImageInfo(depthId, i));
        utils::descriptorSet::update(device, writes);
    }
}

void Graphics::Base::render(uint32_t frameNumber, VkCommandBuffer commandBuffers) const
{
    if (!objects) return;

    uint32_t primitiveCount = 0;
    for(const auto& object: *objects){
        if(!object) continue;

        const auto mask = object->objectMask();
        const auto type = mask.type();
        const auto property = mask.property();
        const auto model = object->model();

        if (!model) continue;
        if (!property.has(interfaces::ObjectProperty::enable)) continue;
        if (!type.has_any(interfaces::ObjectType::Value::baseTypes)) continue;

        const auto& pipelineDesc = pipelineDescs.at(type);

        vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineDesc.pipeline);

        const utils::vkDefault::DescriptorSets descriptors = {descriptorSets.at(frameNumber), object->getDescriptorSet(frameNumber)};

        object->primitiveRange().first = primitiveCount;
        model->render(object->getInstanceNumber(frameNumber), commandBuffers, pipelineDesc.pipelineLayout, descriptors, primitiveCount);
        object->primitiveRange().setLast(primitiveCount);
    }
}

}
