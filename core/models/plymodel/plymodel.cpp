#include "plymodel.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "operations.h"
#include "device.h"

#include <memory>
#include <fstream>
#include <vector>

namespace moon::models {

PlyModel::PlyModel(
        std::filesystem::path filename,
        moon::math::Vector<float, 4> baseColorFactor,
        moon::math::Vector<float, 4> diffuseFactor,
        moon::math::Vector<float, 4> specularFactor,
        float metallicFactor,
        float roughnessFactor,
        float workflow) : filename(filename)
{
    materialBlock.baseColorFactor = baseColorFactor;
    materialBlock.diffuseFactor = diffuseFactor;
    materialBlock.specularFactor = specularFactor;
    materialBlock.metallicFactor = metallicFactor;
    materialBlock.roughnessFactor = roughnessFactor;
    materialBlock.workflow = workflow;
}

moon::interfaces::MaterialBlock &PlyModel::getMaterialBlock(){
    return materialBlock;
}

void PlyModel::destroyCache() {
    vertexCache = utils::Buffer();
    indexCache = utils::Buffer();
}

const VkBuffer* PlyModel::vertexBuffer() const {
    return vertices;
}

const VkBuffer* PlyModel::indexBuffer() const {
    return indices;
}

const moon::math::Vector<float,3> PlyModel::getMaxSize() const {
    return maxSize;
}

void PlyModel::loadFromFile(VkPhysicalDevice physicalDevice, VkDevice device, VkCommandBuffer commandBuffer) {
    tinyply::PlyFile file;
    std::ifstream file_stream(filename, std::ios::binary);
    file.parse_header(file_stream);

    std::shared_ptr<tinyply::PlyData> vertices, normals, texcoords, faces;
    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { texcoords = file.request_properties_from_element("vertex", { "u", "v" });} catch (const std::exception & e) {static_cast<void>(e);}
    try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);} catch (const std::exception & e) {static_cast<void>(e);}

    file.read(file_stream);

    indexCount = faces ? 3 * static_cast<uint32_t>(faces->count) : 0;
    std::vector<uint32_t> indexBuffer(indexCount);
    std::vector<interfaces::Vertex> vertexBuffer(vertices? vertices->count : 0, interfaces::Vertex());

    if(vertices){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < vertices->buffer.size_bytes(); bufferIndex += 3 * sizeof(float), vertexIndex++){
            std::memcpy((void*)&vertexBuffer[vertexIndex].pos, (void*)&vertices->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
        for(uint32_t i = 0; i < vertexBuffer.size(); i++){
            maxSize = moon::math::Vector<float,3>(
                std::max(maxSize[0],std::abs(vertexBuffer[i].pos[0])),
                std::max(maxSize[1],std::abs(vertexBuffer[i].pos[1])),
                std::max(maxSize[2],std::abs(vertexBuffer[i].pos[2]))
                );
            bb.max = moon::math::Vector<float,3>(
                std::max(bb.max[0],vertexBuffer[i].pos[0]),
                std::max(bb.max[1],vertexBuffer[i].pos[1]),
                std::max(bb.max[2],vertexBuffer[i].pos[2])
            );
            bb.min = moon::math::Vector<float,3>(
                std::min(bb.min[0],vertexBuffer[i].pos[0]),
                std::min(bb.min[1],vertexBuffer[i].pos[1]),
                std::min(bb.min[2],vertexBuffer[i].pos[2])
            );
        }
    }
    if(faces){
        for(size_t bufferIndex = 0, index = 0; bufferIndex < faces->buffer.size_bytes(); bufferIndex += sizeof(uint32_t), index++){
            std::memcpy(&indexBuffer[index], &faces->buffer.get()[bufferIndex], sizeof(uint32_t));
        }
    }
    if(normals){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < normals->buffer.size_bytes(); bufferIndex += 3 * sizeof(float), vertexIndex++){
            std::memcpy((void*)&vertexBuffer[vertexIndex].normal, (void*)&normals->buffer.get()[bufferIndex], 3 * sizeof(float));
        }
    } else if(vertices) {
        for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
            const moon::math::Vector<float, 3> n = normalize(cross(
                vertexBuffer[indexBuffer[i + 1]].pos - vertexBuffer[indexBuffer[i + 0]].pos,
                vertexBuffer[indexBuffer[i + 2]].pos - vertexBuffer[indexBuffer[i + 1]].pos
            ));

            vertexBuffer[indexBuffer[i + 0]].normal += n;
            vertexBuffer[indexBuffer[i + 1]].normal += n;
            vertexBuffer[indexBuffer[i + 2]].normal += n;
        }
        for(uint32_t i = 0; i < vertexBuffer.size(); i++){
            vertexBuffer[i].normal = normalize(vertexBuffer[i].normal);
        }
    }
    if(texcoords){
        for(size_t bufferIndex = 0, vertexIndex = 0; bufferIndex < texcoords->buffer.size_bytes(); bufferIndex += 2 * sizeof(float), vertexIndex++){
            std::memcpy((void*)&vertexBuffer[vertexIndex].uv0, (void*)&texcoords->buffer.get()[bufferIndex], 2 * sizeof(float));
        }
    }

    utils::createDeviceBuffer(physicalDevice, device, commandBuffer, vertexBuffer.size() * sizeof(interfaces::Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, this->vertices);
    utils::createDeviceBuffer(physicalDevice, device, commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);

    this->uniformBlock.mat = moon::math::Matrix<float,4,4>(1.0f);
    uniformBuffer = utils::vkDefault::Buffer(physicalDevice, device, sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uniformBuffer.copy(&uniformBlock);
    moon::utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", plyModel::loadFromFile, uniformBuffer");
}

void PlyModel::createDescriptorPool() {
    nodeDescriptorSetLayout = moon::interfaces::Model::createMeshDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &materialDescriptorSetLayout, &nodeDescriptorSetLayout }, 1);
}

void PlyModel::createDescriptorSet() {
    {
        uniformBuffer.descriptorSet = descriptorPool.allocateDescriptorSets(nodeDescriptorSetLayout, 1).front();

        VkDescriptorBufferInfo bufferInfo{ uniformBuffer, 0, sizeof(uniformBlock)};
        VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
    }

    {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = materialDescriptorSetLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material.descriptorSet));

        auto getDescriptorImageInfo = [this](const moon::utils::Texture* tex){
            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView   = tex ? tex->imageView() : emptyTexture.imageView();
            descriptorImageInfo.sampler     = tex ? tex->sampler()   : emptyTexture.sampler();
            return descriptorImageInfo;
        };

        VkDescriptorImageInfo baseColorTextureInfo{};

        VkDescriptorImageInfo metallicRoughnessTextureInfo{};
        switch (material.pbrWorkflows)
        {
        case interfaces::Material::PbrWorkflow::MERALLIC_ROUGHNESS: {
            baseColorTextureInfo = getDescriptorImageInfo(material.baseColor.texture);
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material.metallicRoughness.texture);
            break;
        }
        case interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS: {
            baseColorTextureInfo = getDescriptorImageInfo(material.extensions.diffuse.texture);
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material.extensions.specularGlossiness.texture);
            break;
        }
        }

        std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
            baseColorTextureInfo,
            metallicRoughnessTextureInfo,
            getDescriptorImageInfo(material.normal.texture),
            getDescriptorImageInfo(material.occlusion.texture),
            getDescriptorImageInfo(material.emissive.texture)
        };

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(const auto& info: descriptorImageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = material.descriptorSet;
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size()) - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &info;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void PlyModel::create(const moon::utils::PhysicalDevice& device, VkCommandPool commandPool)
{
    if(this->device)
    {
        CHECK_M(VkPhysicalDevice(device), std::string("[ PlyModel::create ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(VkDevice(device.device()), std::string("[ PlyModel::create ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool, std::string("[ PlyModel::create ] VkCommandPool is VK_NULL_HANDLE"));

        emptyTexture = utils::Texture::empty(device, commandPool);
        this->device = device.device();

        VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.device(), commandPool);
        loadFromFile(device, device.device(), commandBuffer);
        moon::utils::singleCommandBuffer::submit(device.device(), device.device()(0,0), commandPool, &commandBuffer);
        destroyCache();
        createDescriptorPool();
        createDescriptorSet();
    }
}

void PlyModel::render(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const {
    auto descriptors = descriptorSets;
    descriptors.push_back(uniformBuffer.descriptorSet);
    descriptors.push_back(material.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

    auto pushConstants = materialBlock;
    pushConstants.primitive = primitiveCount++;
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushConstants), &pushConstants);

    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
}

void PlyModel::renderBB(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    auto descriptors = descriptorSets;
    descriptors.push_back(uniformBuffer.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(bb), &bb);
    vkCmdDraw(commandBuffer, 24, 1, 0, 0);
}

}
