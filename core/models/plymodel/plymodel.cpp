#include "plymodel.h"

#include <memory>
#include <fstream>
#include <vector>

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <utils/memory.h>
#include <utils/device.h>

#include <math/linearAlgebra.h>

namespace moon::models {

PlyModel::PlyModel(
    const std::filesystem::path& filename,
    const math::vec4& baseColorFactor,
    const math::vec4& diffuseFactor,
    const math::vec4& specularFactor,
    const float metallicFactor,
    const float roughnessFactor,
    const interfaces::Material::PbrWorkflow workflow) :
    filename(filename)
{
    type = interfaces::Model::VertexType::baseSimple;

    auto& mat = materials.emplace_back();
    mat.baseColor.factor = baseColorFactor;
    mat.baseColor.coordSet = 1;
    mat.extensions.diffuse.factor = diffuseFactor;
    mat.extensions.specularGlossiness.factor = specularFactor;
    mat.metallicRoughness.factor[interfaces::Material::metallicIndex] = metallicFactor;
    mat.metallicRoughness.factor[interfaces::Material::roughnessIndex] = roughnessFactor;
    mat.pbrWorkflows = workflow;
}

interfaces::Material& PlyModel::material() {return materials.back();}
const interfaces::Material& PlyModel::material() const { return materials.back(); }
math::box PlyModel::boundingBox() const { return mesh.primitives.empty() ? math::box() : mesh.primitives.back().bb; }

void PlyModel::destroyCache() {
    cache = Cache();
}

bool PlyModel::loadFromFile(const utils::PhysicalDevice& physicalDevice, VkCommandBuffer commandBuffer) {
    std::ifstream file_stream(filename, std::ios::binary);
    tinyply::PlyFile file;

    if(!CHECK_M(file.parse_header(file_stream), "[ PlyModel::loadFromFile ] : fail to parse header")) return false;

    std::shared_ptr<tinyply::PlyData> verts, normals, texcoords, faces, tripstrip;
    try { verts = file.request_properties_from_element("vertex", { "x", "y", "z" });} catch (const std::exception&) {}
    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" });} catch (const std::exception&) {}
    try { texcoords = file.request_properties_from_element("vertex", { "u", "v" });} catch (const std::exception&) {}
    try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3);} catch (const std::exception&) {}
    try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); } catch (const std::exception&) {}

    if (!CHECK_M(!tripstrip.get(), "[ PlyModel::loadFromFile ] : tripstrip unsupported")) return false;

    file.read(file_stream);

    struct Host {
        interfaces::Indices indices;
        interfaces::SimpleVertices vertices;
    };
    Host host = { interfaces::Indices(faces ? 3 * faces->count : 0), interfaces::SimpleVertices(verts ? verts->count : 0) };
    math::box bb;

    if (faces) {
        std::memcpy(host.indices.data(), faces->buffer.get_const(), faces->buffer.size_bytes());
    }

    if(verts){
        const auto buffer = (const math::vec3*)verts->buffer.get();
        for (size_t i = 0; i < host.vertices.size(); ++i) {
            host.vertices[i].pos = buffer[i];
        }
        for (uint32_t i = 0; i < host.indices.size(); i += 3) {
            auto& vert0 = host.vertices[host.indices[i + 0]],
                & vert1 = host.vertices[host.indices[i + 1]],
                & vert2 = host.vertices[host.indices[i + 2]];
            const auto n = normalized(cross(vert1.pos - vert0.pos, vert2.pos - vert1.pos));
            vert0.normal += n; vert1.normal += n; vert2.normal += n;
        }
        for(auto& vert : host.vertices){
            vert.normal = normalized(vert.normal);
            bb.max = math::vec4(math::max(bb.max.dvec(), vert.pos), -1.0f);
            bb.min = math::vec4(math::min(bb.min.dvec(), vert.pos), -1.0f);
        }
    }

    if(normals){
        const auto buffer = (const math::vec3*)normals->buffer.get();
        for (size_t index = 0; index < host.vertices.size(); ++index) {
            host.vertices[index].normal = buffer[index];
        }
    }

    if(texcoords){
        const auto buffer = (const math::vec2*)texcoords->buffer.get();
        for (size_t index = 0; index < host.vertices.size(); ++index) {
            host.vertices[index].uv0 = buffer[index];
        }
    }

    textures.push_back(utils::Texture::empty(physicalDevice, commandBuffer));
    material().baseColor.texture = &textures.back();
    material().metallicRoughness.texture = &textures.back();
    material().normal.texture = &textures.back();
    material().occlusion.texture = &textures.back();
    material().emissive.texture = &textures.back();
    material().extensions.specularGlossiness.texture = &textures.back();
    material().extensions.diffuse.texture = &textures.back();

    mesh.primitives.push_back(
        interfaces::Primitive({0, (uint32_t)host.indices.size()}, {0, (uint32_t)host.vertices.size()}, &material(), bb)
    );

    const auto& device = physicalDevice.device();
    utils::createDeviceBuffer(physicalDevice, device, commandBuffer, host.vertices.size() * sizeof(decltype(host.vertices)::value_type), host.vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, cache.vertices, vertices);
    if (!host.indices.empty()) {
        utils::createDeviceBuffer(physicalDevice, device, commandBuffer, host.indices.size() * sizeof(decltype(host.indices)::value_type), host.indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, cache.indices, indices);
    }

    skeleton.deviceBuffer = utils::vkDefault::Buffer(physicalDevice, device, sizeof(math::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    utils::Memory::instance().nameMemory(skeleton.deviceBuffer, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", plyModel::loadFromFile, uniformBuffer");
    skeleton.deviceBuffer.copy(&skeleton.hostBuffer);

    return true;
}

void PlyModel::createDescriptors(VkDevice device) {
    skeletonDescriptorSetLayout = interfaces::Skeleton::descriptorSetLayout(device);
    materialDescriptorSetLayout = interfaces::Material::descriptorSetLayout(device);
    descriptorPool = utils::vkDefault::DescriptorPool(device, { &materialDescriptorSetLayout, &skeletonDescriptorSetLayout }, 1);

    skeleton.createDescriptorSet(device, descriptorPool, skeletonDescriptorSetLayout);
    material().createDescriptorSet(device, descriptorPool, materialDescriptorSetLayout);
}

void PlyModel::create(const utils::PhysicalDevice& device, VkCommandPool commandPool)
{
    if(
        CHECK_M(VkPhysicalDevice(device), "[ PlyModel::create ] VkPhysicalDevice is VK_NULL_HANDLE") &&
        CHECK_M(VkDevice(device.device()), "[ PlyModel::create ] VkDevice is VK_NULL_HANDLE") &&
        CHECK_M(commandPool, "[ PlyModel::create ] VkCommandPool is VK_NULL_HANDLE")
    ) {
        utils::singleCommandBuffer::Scoped commandBuffer(device.device(), device.device()(0, 0), commandPool);
        if (loadFromFile(device, commandBuffer)) {
            createDescriptors(device.device());
        }
    }
    destroyCache();
}

void PlyModel::render(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets, uint32_t& primitiveCount) const {
    VkDeviceSize offsets = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertices, &offsets);
    if (VkBuffer(indices) != VK_NULL_HANDLE) {
        vkCmdBindIndexBuffer(commandBuffer, indices, 0, VK_INDEX_TYPE_UINT32);
    }
    auto descriptors = descriptorSets;
    descriptors.push_back(skeleton.descriptorSet);
    mesh.render(commandBuffer, pipelineLayout, descriptors, primitiveCount);
}

void PlyModel::renderBB(uint32_t, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, const utils::vkDefault::DescriptorSets& descriptorSets) const {
    auto descriptors = descriptorSets;
    descriptors.push_back(skeleton.descriptorSet);
    mesh.renderBB(commandBuffer, pipelineLayout, descriptors);
}

std::vector<interfaces::Animation*> PlyModel::animations(uint32_t instanceNumber) {
    return {};
}

}
