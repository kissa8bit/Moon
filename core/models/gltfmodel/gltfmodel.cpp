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

void GltfModel::loadFromFile(const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
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
    nodeDescriptorSetLayout = interfaces::Model::createMeshDescriptorSetLayout(device);
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

void GltfModel::render(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t &primitiveCount) const {
    for (auto& [_, node] : instances.at(instanceNumber).nodes) {
        if (!CHECK_M(node.get(), std::string("[ GltfModel::render ] node is nullptr"))) continue;
        for (const Primitive& primitive : node->mesh.primitives) {
            auto descriptors = descriptorSets;
            descriptors.push_back(node->mesh.descriptorSet);
            descriptors.push_back(primitive.material->descriptorSet);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

            if (!CHECK_M(primitive.material, std::string("[ GltfModel::render ] material is nullptr"))) continue;
            const auto& material = *primitive.material;

            interfaces::MaterialBlock materialBlock{};
            materialBlock.primitive = primitiveCount++;
            materialBlock.emissiveFactor = material.emissive.factor;
            materialBlock.colorTextureSet = material.baseColor.coordSet;
            materialBlock.normalTextureSet = material.normal.coordSet;
            materialBlock.occlusionTextureSet = material.occlusion.coordSet;
            materialBlock.emissiveTextureSet = material.emissive.coordSet;
            materialBlock.alphaMask = static_cast<float>(material.alphaMode == interfaces::Material::ALPHAMODE_MASK);
            materialBlock.alphaMaskCutoff = material.alphaCutoff;

            switch (material.pbrWorkflows)
            {
                case interfaces::Material::PbrWorkflow::MERALLIC_ROUGHNESS : {
                    materialBlock.workflow = static_cast<float>(interfaces::Material::PbrWorkflow::MERALLIC_ROUGHNESS);
                    materialBlock.baseColorFactor = material.baseColor.factor;
                    materialBlock.metallicFactor = material.metallicRoughness.factor[0];
                    materialBlock.roughnessFactor = material.metallicRoughness.factor[1];
                    materialBlock.physicalDescriptorTextureSet = material.metallicRoughness.coordSet;
                    break;
                }
                case interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS: {
                    materialBlock.workflow = static_cast<float>(interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS);
                    materialBlock.physicalDescriptorTextureSet = material.extensions.specularGlossiness.coordSet;
                    materialBlock.diffuseFactor = material.extensions.diffuse.factor;
                    materialBlock.specularFactor = material.extensions.specularGlossiness.factor;
                    break;
                }
            }

            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(materialBlock), &materialBlock);

            if (primitive.indexCount > 0) {
                vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
            }
            else {
                vkCmdDraw(commandBuffer, primitive.vertexCount, 1, 0, 0);
            }
        }
    }
}

void GltfModel::renderBB(uint32_t instanceNumber, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    for (auto& [_, node] : instances.at(instanceNumber).nodes) {
        for (const Primitive& primitive : node->mesh.primitives) {
            auto descriptors = descriptorSets;
            descriptors.push_back(node->mesh.descriptorSet);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(interfaces::BoundingBox), (void*)&primitive.bb);
            vkCmdDraw(commandBuffer, 24, 1, 0, 0);
        }
    }
}

}
