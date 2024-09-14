#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include <cstring>

#include "operations.h"
#include "device.h"

#include "gltfmodel.h"
#include "gltfmodel/gltfutils.h"
#include "gltfmodel/node.h"

namespace moon::models {

namespace {

void calculateTangent(std::vector<interfaces::Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer){
    for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
        const auto &v0 = vertexBuffer[indexBuffer[i + 0]], &v1 = vertexBuffer[indexBuffer[i + 1]], &v2 = vertexBuffer[indexBuffer[i + 2]];

        const auto dv1 = v1.pos - v0.pos;
        const auto dv2 = v2.pos - v0.pos;
        const auto duv1 = v1.uv0 - v0.uv0;
        const auto duv2 = v2.uv0 - v0.uv0;

        const float det = 1.0f / (duv1[0] * duv2[1] - duv1[1] * duv2[0]);
        const auto bitangent = normalize( det * (duv1[0] * dv2 - duv2[0] * dv1));
        auto tangent = normalize(det * (duv2[1] * dv1 - duv1[1] * dv2));

        if(dot(cross(tangent, bitangent), v0.normal) < 0.0f){
            tangent = -1.0f * tangent;
        }

        for(uint32_t j = i; j < i + 3; j++){
            auto& v = vertexBuffer[indexBuffer[j]];
            v.tangent = normalize(tangent - v.normal * dot(v.normal, tangent));
            v.bitangent = normalize(cross(v.normal, v.tangent));
        }
    }
}

void renderNode(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant)
{
    for (const Primitive& primitive : node->mesh.primitives)
    {
        utils::vkDefault::DescriptorSets nodeDescriptorSets(descriptorSetsCount);
        std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
        nodeDescriptorSets.push_back(node->mesh.descriptorSet);
        nodeDescriptorSets.push_back(primitive.material->descriptorSet);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 2, nodeDescriptorSets.data(), 0, NULL);

        interfaces::MaterialBlock material{};
        material.primitive = primitiveCount++;
        material.emissiveFactor = primitive.material->emissiveFactor;
        material.colorTextureSet = primitive.material->texCoordSets.baseColor;
        material.normalTextureSet = primitive.material->texCoordSets.normal;
        material.occlusionTextureSet = primitive.material->texCoordSets.occlusion;
        material.emissiveTextureSet = primitive.material->texCoordSets.emissive;
        material.alphaMask = static_cast<float>(primitive.material->alphaMode == interfaces::Material::ALPHAMODE_MASK);
        material.alphaMaskCutoff = primitive.material->alphaCutoff;
        if (primitive.material->pbrWorkflows.metallicRoughness) {
            material.workflow = static_cast<float>(interfaces::PBR_WORKFLOW_METALLIC_ROUGHNESS);
            material.baseColorFactor = primitive.material->baseColorFactor;
            material.metallicFactor = primitive.material->metallicFactor;
            material.roughnessFactor = primitive.material->roughnessFactor;
            material.PhysicalDescriptorTextureSet = primitive.material->texCoordSets.metallicRoughness;
            material.colorTextureSet = primitive.material->texCoordSets.baseColor;
        }
        if (primitive.material->pbrWorkflows.specularGlossiness) {
            material.workflow = static_cast<float>(interfaces::PBR_WORKFLOW_SPECULAR_GLOSINESS);
            material.PhysicalDescriptorTextureSet = primitive.material->texCoordSets.specularGlossiness;
            material.colorTextureSet = primitive.material->texCoordSets.baseColor;
            material.diffuseFactor = primitive.material->extension.diffuseFactor;
            material.specularFactor = math::Vector<float, 4>(
                primitive.material->extension.specularFactor[0],
                primitive.material->extension.specularFactor[1],
                primitive.material->extension.specularFactor[2],
                1.0f);
        }
        std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &material, sizeof(interfaces::MaterialBlock));

        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

        if (primitive.indexCount > 0) {
            vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
        } else {
            vkCmdDraw(commandBuffer, primitive.vertexCount, 1, 0, 0);
        }
    }
}

void renderNodeBB(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
{
    for (const Primitive& primitive : node->mesh.primitives) {
        utils::vkDefault::DescriptorSets nodeDescriptorSets(descriptorSetsCount);
        std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
        nodeDescriptorSets.push_back(node->mesh.descriptorSet);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 1, nodeDescriptorSets.data(), 0, NULL);

        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(interfaces::BoundingBox), (void*)&primitive.bb);
        vkCmdDraw(commandBuffer, 24, 1, 0, 0);
    }
}

}

GltfModel::GltfModel(std::filesystem::path filename, uint32_t instanceCount) : filename(filename) {
    instances.resize(instanceCount);
}

void GltfModel::destroyCache() {
    for(auto& texture: textures) texture.destroyCache();
    vertexCache = utils::Buffer();
    indexCache = utils::Buffer();
}

const VkBuffer* GltfModel::vertexBuffer() const {
    return vertices;
}

const VkBuffer* GltfModel::indexBuffer() const {
    return indices;
}

void GltfModel::loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer)
{
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;

    std::string error{}, warning{};
    const auto loadFileMethod = filename.extension() == ".glb" ? &tinygltf::TinyGLTF::LoadBinaryFromFile : &tinygltf::TinyGLTF::LoadASCIIFromFile;
    const auto loadSuccess = (gltfContext.*loadFileMethod)(&gltfModel, &error, &warning, filename.string(), tinygltf::SectionCheck::REQUIRE_VERSION);

    if (!loadSuccess) return;

    loadTextures(gltfModel, device, commandBuffer);
    loadMaterials(gltfModel);

    for(auto& instance: instances){
        uint32_t indexStart = 0;
        for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
            loadNode(gltfModel, device, instance.nodes, nullptr, nodeIndex, indexStart);
        }
    }

    std::vector<uint32_t> indexBuffer;
    std::vector<interfaces::Vertex> vertexBuffer;
    for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
        loadVertexBuffer(gltfModel, gltfModel.nodes[nodeIndex], indexBuffer, vertexBuffer);
    }
    calculateTangent(vertexBuffer, indexBuffer);

    loadSkins(gltfModel);
    if (gltfModel.animations.size() > 0) {
        loadAnimations(gltfModel);
    }

    for(auto& instance : instances){
        for (auto& [_, node] : instance.nodes) {
            node->update();
        }
    }

    utils::createDeviceBuffer(device, device.device(), commandBuffer, vertexBuffer.size() * sizeof(interfaces::Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, vertices);
    utils::createDeviceBuffer(device, device.device(), commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);
}

void GltfModel::createDescriptors(VkDevice device) {
    nodeDescriptorSetLayout = interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = interfaces::Model::createMaterialDescriptorSetLayout(device);

    std::vector<const utils::vkDefault::DescriptorSetLayout*> layouts(materials.size(), &materialDescriptorSetLayout);
    for (const auto& instance : instances) {
        for (const auto& _ : instance.nodes) {
            layouts.push_back(&nodeDescriptorSetLayout);
        }
    }

    descriptorPool = utils::vkDefault::DescriptorPool(device, layouts, 1);

    for (auto& instance : instances) {
        for (auto& [_, node] : instance.nodes) {
            node->mesh.createDescriptorSet(device, descriptorPool, nodeDescriptorSetLayout);
        }
    }

    for (auto& material : materials) {
        material.createDescriptorSet(device, descriptorPool, materialDescriptorSetLayout);
    }
}

void GltfModel::create(const utils::PhysicalDevice& device, VkCommandPool commandPool) {
    if(
        CHECK_M(VkPhysicalDevice(device), std::string("[ GltfModel::create ] VkPhysicalDevice is VK_NULL_HANDLE")) &&
        CHECK_M(VkDevice(device.device()), std::string("[ GltfModel::create ] VkDevice is VK_NULL_HANDLE")) &&
        CHECK_M(commandPool, std::string("[ GltfModel::create ] VkCommandPool is VK_NULL_HANDLE"))
    ) {
        utils::singleCommandBuffer::Scoped commandBuffer(device.device(), device.device()(0, 0), commandPool);
        loadFromFile(device, commandBuffer);
        createDescriptors(device.device());
    }
    destroyCache();
}

void GltfModel::render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t &primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant){
    for (auto& [_, node] : instances[frameIndex].nodes) {
        renderNode(node.get(), commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

void GltfModel::renderBB(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets){
    for (auto& [_, node] : instances[frameIndex].nodes) {
        renderNodeBB(node.get(), commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets);
    }
}

}
