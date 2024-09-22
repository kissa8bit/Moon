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
    const std::filesystem::path& filename,
    const math::Vector<float, 4>& baseColorFactor,
    const math::Vector<float, 4>& diffuseFactor,
    const math::Vector<float, 4>& specularFactor,
    const float metallicFactor,
    const float roughnessFactor,
    const const interfaces::Material::PbrWorkflow workflow) : filename(filename)
{
    mat.baseColor.factor = baseColorFactor;
    mat.extensions.diffuse.factor = diffuseFactor;
    mat.extensions.specularGlossiness.factor = specularFactor;
    mat.metallicRoughness.factor[interfaces::Material::metallicIndex] = metallicFactor;
    mat.metallicRoughness.factor[interfaces::Material::roughnessIndex] = roughnessFactor;
    mat.pbrWorkflows = workflow;
}

interfaces::Material& PlyModel::material() {return mat;}
interfaces::BoundingBox& PlyModel::boundingBox() { return bb; }

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

void PlyModel::loadFromFile(const utils::PhysicalDevice& physicalDevice, VkCommandBuffer commandBuffer) {
    std::ifstream file_stream(filename, std::ios::binary);
    tinyply::PlyFile file;

    if(!CHECK_M(file.parse_header(file_stream), "[ PlyModel::loadFromFile ] : fail to parse header")) return;

    std::shared_ptr<tinyply::PlyData> verts, normals, texcoords, faces, tripstrip;
    try { verts = file.request_properties_from_element("vertex", { "x", "y", "z" });} catch (const std::exception&) {}
    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" });} catch (const std::exception&) {}
    try { texcoords = file.request_properties_from_element("vertex", { "u", "v" });} catch (const std::exception&) {}
    try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);} catch (const std::exception&) {}
    try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); } catch (const std::exception&) {}

    if (!CHECK_M(!tripstrip.get(), "[ PlyModel::loadFromFile ] : tripstrip unsupported")) return;

    file.read(file_stream);

    vertexCount = verts->count;
    indexCount = faces ? 3 * faces->count : 0;
    std::vector<uint32_t> indexBuffer(indexCount);
    std::vector<interfaces::Vertex> vertexBuffer(verts ? verts->count : 0, interfaces::Vertex());

    if (faces) {
        std::memcpy(indexBuffer.data(), faces->buffer.get_const(), faces->buffer.size_bytes());
    }
    if(verts){
        const auto buffer = (const math::Vector<float, 3>*)verts->buffer.get();
        for (size_t i = 0; i < vertexBuffer.size(); ++i) {
            vertexBuffer[i].pos = buffer[i];
        }
        for (uint32_t i = 0; i < indexBuffer.size(); i += 3) {
            auto& vert0 = vertexBuffer[indexBuffer[i + 0]], & vert1 = vertexBuffer[indexBuffer[i + 1]], & vert2 = vertexBuffer[indexBuffer[i + 2]];
            const auto n = normalize(cross(vert1.pos - vert0.pos, vert2.pos - vert1.pos));
            vert0.normal += n; vert1.normal += n; vert2.normal += n;
        }
        for(auto& vert : vertexBuffer){
            vert.normal = normalize(vert.normal);
            bb.max = math::maximum(bb.max, vert.pos);
            bb.min = math::minimum(bb.min, vert.pos);
        }
    }
    if(normals){
        const auto buffer = (const math::Vector<float, 3>*)normals->buffer.get();
        for (size_t index = 0; index < vertexBuffer.size(); ++index) {
            vertexBuffer[index].normal = buffer[index];
        }
    }
    if(texcoords){
        const auto buffer = (const math::Vector<float, 2>*)texcoords->buffer.get();
        for (size_t index = 0; index < vertexBuffer.size(); ++index) {
            vertexBuffer[index].uv0 = buffer[index];
        }
    }

    textures.push_back(utils::Texture::empty(physicalDevice, commandBuffer));
    mat.baseColor.texture = &textures.back();
    mat.metallicRoughness.texture = &textures.back();
    mat.normal.texture = &textures.back();
    mat.occlusion.texture = &textures.back();
    mat.emissive.texture = &textures.back();
    mat.extensions.specularGlossiness.texture = &textures.back();
    mat.extensions.diffuse.texture = &textures.back();

    const auto& device = physicalDevice.device();
    utils::createDeviceBuffer(physicalDevice, device, commandBuffer, vertexBuffer.size() * sizeof(interfaces::Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, vertices);
    if (indexCount > 0) {
        utils::createDeviceBuffer(physicalDevice, device, commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);
    }

    interfaces::MeshBlock uniformBlock{};
    uniformBuffer = utils::vkDefault::Buffer(physicalDevice, device, sizeof(uniformBlock), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    utils::Memory::instance().nameMemory(uniformBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", plyModel::loadFromFile, uniformBuffer");
    uniformBuffer.copy(&uniformBlock);
}

void PlyModel::createDescriptors(VkDevice device) {
    meshDescriptorSetLayout = interfaces::Model::createMeshDescriptorSetLayout(device);
    materialDescriptorSetLayout = interfaces::Model::createMaterialDescriptorSetLayout(device);
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &materialDescriptorSetLayout, &meshDescriptorSetLayout }, 1);

    descriptorSet = descriptorPool.allocateDescriptorSets(meshDescriptorSetLayout, 1).front();
    VkDescriptorBufferInfo bufferInfo{ uniformBuffer, 0, sizeof(interfaces::MeshBlock) };
    VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

    mat.createDescriptorSet(device, descriptorPool, materialDescriptorSetLayout);
}

void PlyModel::create(const utils::PhysicalDevice& device, VkCommandPool commandPool)
{
    if(
        CHECK_M(VkPhysicalDevice(device), std::string("[ PlyModel::create ] VkPhysicalDevice is VK_NULL_HANDLE")) &&
        CHECK_M(VkDevice(device.device()), std::string("[ PlyModel::create ] VkDevice is VK_NULL_HANDLE")) &&
        CHECK_M(commandPool, std::string("[ PlyModel::create ] VkCommandPool is VK_NULL_HANDLE"))
    ) {
        utils::singleCommandBuffer::Scoped commandBuffer(device.device(), device.device()(0, 0), commandPool);
        loadFromFile(device, commandBuffer);
        createDescriptors(device.device());
    }
    destroyCache();
}

void PlyModel::render(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const {
    auto descriptors = descriptorSets;
    descriptors.push_back(descriptorSet);
    descriptors.push_back(mat.descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

    interfaces::MaterialBlock materialBlock(mat, primitiveCount++);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(materialBlock), &materialBlock);

    if (indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
    }
    else {
        vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
    }
}

void PlyModel::renderBB(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    auto descriptors = descriptorSets;
    descriptors.push_back(descriptorSet);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptors.size(), descriptors.data(), 0, NULL);

    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(bb), &bb);
    vkCmdDraw(commandBuffer, 24, 1, 0, 0);
}

}
