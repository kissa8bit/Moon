#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STBI_MSC_SECURE_CRT

#include "gltfmodel.h"
#include "operations.h"
#include "device.h"

#include <cstring>

namespace moon::models {

namespace {

    VkSamplerAddressMode getVkWrapMode(int32_t wrapMode){
        switch (wrapMode) {
        case 10497:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 33071:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 33648:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        };
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }

    VkFilter getVkFilterMode(int32_t filterMode){
        switch (filterMode) {
        case 9728:
        case 9984:
        case 9985:
            return VK_FILTER_NEAREST;
        case 9729:
        case 9986:
        case 9987:
            return VK_FILTER_LINEAR;
        }
        return VK_FILTER_LINEAR;
    }

    void calculateNodeTangent(std::vector<moon::interfaces::Model::Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer){
        for(uint32_t i = 0; i < indexBuffer.size(); i += 3){
            moon::math::Vector<float,3> dv1   = vertexBuffer[indexBuffer[i+1]].pos - vertexBuffer[indexBuffer[i+0]].pos;
            moon::math::Vector<float,3> dv2   = vertexBuffer[indexBuffer[i+2]].pos - vertexBuffer[indexBuffer[i+0]].pos;

            moon::math::Vector<float,2> duv1  = vertexBuffer[indexBuffer[i+1]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;
            moon::math::Vector<float,2> duv2  = vertexBuffer[indexBuffer[i+2]].uv0 - vertexBuffer[indexBuffer[i+0]].uv0;

            float det = 1.0f/(duv1[0]*duv2[1] - duv1[1]*duv2[0]);

            moon::math::Vector<float,3> tangent = normalize( det*(duv2[1]*dv1-duv1[1]*dv2));
            moon::math::Vector<float,3> bitangent = normalize( det*(duv1[0]*dv2-duv2[0]*dv1));

            if(dot(cross(tangent,bitangent),vertexBuffer[indexBuffer[i+0]].normal)<0.0f){
                tangent = -1.0f * tangent;
            }

            for(uint32_t index = 0; index < 3; index++){
                vertexBuffer[indexBuffer[i+index]].tangent      = normalize(tangent - vertexBuffer[indexBuffer[i+index]].normal * dot(vertexBuffer[indexBuffer[i+index]].normal, tangent));
                vertexBuffer[indexBuffer[i+index]].bitangent    = normalize(cross(vertexBuffer[indexBuffer[i+index]].normal, vertexBuffer[indexBuffer[i+index]].tangent));
            }
        }
    }

    bool isBinary(const std::filesystem::path& filename){
        size_t extpos = filename.string().rfind('.', filename.string().length());
        return (extpos != std::string::npos) && (filename.string().substr(extpos + 1, filename.string().length() - extpos) == "glb");
    }

    void createNodeDescriptorSet(VkDevice device, Node* node, utils::vkDefault::DescriptorPool& descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout)
    {
        if (node->mesh){
            node->mesh->uniformBuffer.descriptorSet = descriptorPool.allocateDescriptorSets(descriptorSetLayout, 1).front();

            VkDescriptorBufferInfo bufferInfo{ node->mesh->uniformBuffer, 0, sizeof(Mesh::uniformBlock)};
            VkWriteDescriptorSet writeDescriptorSet{};
                writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSet.descriptorCount = 1;
                writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
                writeDescriptorSet.dstBinding = 0;
                writeDescriptorSet.pBufferInfo = &bufferInfo;
            vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        }
        for (auto& child : node->children){
            createNodeDescriptorSet(device,child,descriptorPool,descriptorSetLayout);
        }
    }

    void createMaterialDescriptorSet(VkDevice device, moon::interfaces::Material* material, utils::vkDefault::DescriptorPool & descriptorPool, const utils::vkDefault::DescriptorSetLayout& descriptorSetLayout)
    {
        material->descriptorSet = descriptorPool.allocateDescriptorSets(descriptorSetLayout, 1).front();

        auto getDescriptorImageInfo = [](const moon::utils::Texture* tex){
            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptorImageInfo.imageView = tex->imageView();
            descriptorImageInfo.sampler = tex->sampler();
            return descriptorImageInfo;
        };

        VkDescriptorImageInfo baseColorTextureInfo{};
        if (material->pbrWorkflows.metallicRoughness){
            baseColorTextureInfo = getDescriptorImageInfo(material->baseColorTexture);
        }
        if(material->pbrWorkflows.specularGlossiness){
            baseColorTextureInfo = getDescriptorImageInfo(material->extension.diffuseTexture);
        }

        VkDescriptorImageInfo metallicRoughnessTextureInfo{};
        if (material->pbrWorkflows.metallicRoughness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material->metallicRoughnessTexture);
        }
        if (material->pbrWorkflows.specularGlossiness){
            metallicRoughnessTextureInfo = getDescriptorImageInfo(material->extension.specularGlossinessTexture);
        }

        std::vector<VkDescriptorImageInfo> descriptorImageInfos = {
            baseColorTextureInfo,
            metallicRoughnessTextureInfo,
            getDescriptorImageInfo(material->normalTexture),
            getDescriptorImageInfo(material->occlusionTexture),
            getDescriptorImageInfo(material->emissiveTexture)
        };

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        for(const auto& info: descriptorImageInfos){
            descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = material->descriptorSet;
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size()) - 1;
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &info;
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }

    void renderNode(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t& primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant)
    {
        if (node->mesh)
        {
            for (Mesh::Primitive* primitive : node->mesh->primitives)
            {
                utils::vkDefault::DescriptorSets nodeDescriptorSets(descriptorSetsCount);
                std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
                nodeDescriptorSets.push_back(node->mesh->uniformBuffer.descriptorSet);
                nodeDescriptorSets.push_back(primitive->material->descriptorSet);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount+2, nodeDescriptorSets.data(), 0, NULL);

                moon::interfaces::MaterialBlock material{};
                material.primitive = primitiveCount++;
                material.emissiveFactor = primitive->material->emissiveFactor;
                material.colorTextureSet = primitive->material->texCoordSets.baseColor;
                material.normalTextureSet = primitive->material->texCoordSets.normal;
                material.occlusionTextureSet = primitive->material->texCoordSets.occlusion;
                material.emissiveTextureSet = primitive->material->texCoordSets.emissive;
                material.alphaMask = static_cast<float>(primitive->material->alphaMode == moon::interfaces::Material::ALPHAMODE_MASK);
                material.alphaMaskCutoff = primitive->material->alphaCutoff;
                if (primitive->material->pbrWorkflows.metallicRoughness) {
                    material.workflow = static_cast<float>(moon::interfaces::PBR_WORKFLOW_METALLIC_ROUGHNESS);
                    material.baseColorFactor = primitive->material->baseColorFactor;
                    material.metallicFactor = primitive->material->metallicFactor;
                    material.roughnessFactor = primitive->material->roughnessFactor;
                    material.PhysicalDescriptorTextureSet = primitive->material->texCoordSets.metallicRoughness;
                    material.colorTextureSet = primitive->material->texCoordSets.baseColor;
                }
                if (primitive->material->pbrWorkflows.specularGlossiness) {
                    material.workflow = static_cast<float>(moon::interfaces::PBR_WORKFLOW_SPECULAR_GLOSINESS);
                    material.PhysicalDescriptorTextureSet = primitive->material->texCoordSets.specularGlossiness;
                    material.colorTextureSet = primitive->material->texCoordSets.baseColor;
                    material.diffuseFactor = primitive->material->extension.diffuseFactor;
                    material.specularFactor = moon::math::Vector<float, 4>(
                        primitive->material->extension.specularFactor[0],
                        primitive->material->extension.specularFactor[1],
                        primitive->material->extension.specularFactor[2],
                        1.0f);
                }
                std::memcpy(reinterpret_cast<char*>(pushConstant) + pushConstantOffset, &material, sizeof(moon::interfaces::MaterialBlock));

                vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, pushConstantSize, pushConstant);

                if (primitive->indexCount > 0){
                    vkCmdDrawIndexed(commandBuffer, primitive->indexCount, 1, primitive->firstIndex, 0, 0);
                }else{
                    vkCmdDraw(commandBuffer, primitive->vertexCount, 1, 0, 0);
                }
            }
        }
        for (auto child : node->children){
            renderNode(child, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
        }
    }

    void renderNodeBB(Node *node, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets)
    {
        if (node->mesh)
        {
            for (Mesh::Primitive* primitive : node->mesh->primitives)
            {
                utils::vkDefault::DescriptorSets nodeDescriptorSets(descriptorSetsCount);
                std::copy(descriptorSets, descriptorSets + descriptorSetsCount, nodeDescriptorSets.data());
                nodeDescriptorSets.push_back(node->mesh->uniformBuffer.descriptorSet);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, descriptorSetsCount + 1, nodeDescriptorSets.data(), 0, NULL);

                vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(moon::interfaces::BoundingBox), (void*) &primitive->bb);
                vkCmdDraw(commandBuffer, 24, 1, 0, 0);
            }
        }
        for (auto child : node->children){
            renderNodeBB(child, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets);
        }
    }
}

GltfModel::GltfModel(std::filesystem::path filename, uint32_t instanceCount)
    : filename(filename){
    instances.resize(instanceCount);
}

GltfModel::~GltfModel() {
    for (auto& instance : instances) {
        for (auto& node : instance.nodes) {
            node->destroy(device);
            delete node;
        }
        for (auto& skin : instance.skins) {
            delete skin;
        }
        instance.nodes.clear();
        instance.skins.clear();
        instance.animations.clear();
    }
    textures.clear();
    materials.clear();

    this->device = VK_NULL_HANDLE;
}

void GltfModel::destroyCache()
{
    for(auto& texture: textures) texture.destroyCache();
    vertexCache = utils::Buffer();
    indexCache = utils::Buffer();
}

const VkBuffer* GltfModel::getVertices() const{
    return vertices;
}

const VkBuffer* GltfModel::getIndices() const{
    return indices;
}

void GltfModel::loadSkins(const tinygltf::Model &gltfModel){
    for(auto& instance : instances){
        for (const tinygltf::Skin& source: gltfModel.skins) {
            Skin* newSkin = new Skin{};

            for (int jointIndex : source.joints) {
                if (Node* node = nodeFromIndex(jointIndex, instance.nodes); node) {
                    newSkin->joints.push_back(node);
                }
            }

            if (source.inverseBindMatrices > -1) {
                const tinygltf::Accessor& accessor = gltfModel.accessors[source.inverseBindMatrices];
                const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& buffer = gltfModel.buffers[bufferView.buffer];

                newSkin->inverseBindMatrices.resize(accessor.count);
                std::memcpy((void*)newSkin->inverseBindMatrices.data(), (void*)&buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(moon::math::Matrix<float,4,4>));
                for(auto& matrix: newSkin->inverseBindMatrices){
                    matrix = transpose(matrix);
                }
            }

            for (const auto& node: gltfModel.nodes) {
                if(node.skin == &source - &gltfModel.skins[0]){
                    nodeFromIndex(static_cast<uint32_t>(&node - &gltfModel.nodes[0]), instance.nodes)->skin = newSkin;
                }
            }

            instance.skins.push_back(newSkin);
        }
    }
}

void GltfModel::loadTextures(const moon::utils::PhysicalDevice& device, VkCommandBuffer commandBuffer, const tinygltf::Model& gltfModel)
{
    for(const tinygltf::Texture &tex : gltfModel.textures){
        const tinygltf::Image& gltfimage = gltfModel.images[tex.source];

        const uint32_t downsampleWidth = 1;
        const uint32_t downsampleHeight = 1;

        const auto& width = gltfimage.width / downsampleWidth;
        const auto& height = gltfimage.height / downsampleHeight;

        std::vector<uint8_t> buffer(4 * width * height);

        uint32_t offset = 0;
        for (uint32_t i = 0; i < height; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                uint32_t line = gltfimage.component * (gltfimage.width * downsampleHeight * i + downsampleHeight * j);
                buffer[offset + 3] = 255;
                for (uint32_t k = 0; k < gltfimage.component; ++k) {
                    buffer[offset++] = gltfimage.image[line + k];
                }
            }
        }

        moon::utils::TextureSampler textureSampler{};
        if (tex.sampler != -1) {
            textureSampler.minFilter = textureSampler.magFilter = getVkFilterMode(gltfModel.samplers[tex.sampler].minFilter);
            textureSampler.addressModeV = textureSampler.addressModeW = getVkWrapMode(gltfModel.samplers[tex.sampler].wrapT);
            textureSampler.addressModeU = getVkWrapMode(gltfModel.samplers[tex.sampler].wrapS);
        }
        textures.emplace_back(device, device.device(), commandBuffer, width, height, buffer.data(), textureSampler);
    }
    textures.push_back(utils::Texture::empty(device, commandBuffer));
}

void GltfModel::loadMaterials(const tinygltf::Model &gltfModel)
{
    const utils::Texture* emptyTexture = &textures.back();
    for (const tinygltf::Material &mat : gltfModel.materials)
    {
        moon::interfaces::Material material(emptyTexture);
        if (mat.values.find("baseColorTexture") != mat.values.end()) {
            const auto& baseColor = mat.values.at("baseColorTexture");
            material.baseColorTexture = &textures[baseColor.TextureIndex()];
            material.texCoordSets.baseColor = baseColor.TextureTexCoord();
        }
        if (mat.values.find("metallicRoughnessTexture") != mat.values.end()) {
            const auto& metallicRoughness = mat.values.at("metallicRoughnessTexture");
            material.metallicRoughnessTexture = &textures[metallicRoughness.TextureIndex()];
            material.texCoordSets.metallicRoughness = metallicRoughness.TextureTexCoord();
        }
        if (mat.values.find("roughnessFactor") != mat.values.end()) {
            const auto& roughnessFactor = mat.values.at("roughnessFactor");
            material.roughnessFactor = static_cast<float>(roughnessFactor.Factor());
        }
        if (mat.values.find("metallicFactor") != mat.values.end()) {
            const auto& metallicFactor = mat.values.at("metallicFactor");
            material.metallicFactor = static_cast<float>(metallicFactor.Factor());
        }
        if (mat.values.find("baseColorFactor") != mat.values.end()) {
            const auto& factor = mat.values.at("baseColorFactor").ColorFactor();
            material.baseColorFactor = moon::math::Vector<float, 4>(factor[0], factor[1], factor[2], factor[3]);
        }
        if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end()) {
            const auto& normalTexture = mat.additionalValues.at("normalTexture");
            material.normalTexture = &textures[normalTexture.TextureIndex()];
            material.texCoordSets.normal = normalTexture.TextureTexCoord();
        }
        if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end()) {
            const auto& emissiveTexture = mat.additionalValues.at("emissiveTexture");
            material.emissiveTexture = &textures[emissiveTexture.TextureIndex()];
            material.texCoordSets.emissive = emissiveTexture.TextureTexCoord();
        }
        if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end()) {
            const auto& occlusionTexture = mat.additionalValues.at("occlusionTexture");
            material.occlusionTexture = &textures[occlusionTexture.TextureIndex()];
            material.texCoordSets.occlusion = occlusionTexture.TextureTexCoord();
        }
        if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end()) {
            const tinygltf::Parameter& param = mat.additionalValues.at("alphaMode");
            if (param.string_value == "BLEND") {
                material.alphaMode = moon::interfaces::Material::ALPHAMODE_BLEND;
            }
            if (param.string_value == "MASK") {
                material.alphaCutoff = 0.5f;
                material.alphaMode = moon::interfaces::Material::ALPHAMODE_MASK;
            }
        }
        if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end()) {
            const auto& alphaCutoff = mat.additionalValues.at("alphaCutoff");
            material.alphaCutoff = static_cast<float>(alphaCutoff.Factor());
        }
        if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end()) {
            const auto& factor = mat.additionalValues.at("emissiveFactor").ColorFactor();
            material.emissiveFactor = moon::math::Vector<float, 4>(factor[0], factor[1], factor[2], 1.0f);
        }

        // Extensions
        // @TODO: Find out if there is a nicer way of reading these properties with recent tinygltf headers
        if (mat.extensions.find("KHR_materials_pbrSpecularGlossiness") != mat.extensions.end()) {
            auto ext = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
            if (ext->second.Has("specularGlossinessTexture")) {
                auto index = ext->second.Get("specularGlossinessTexture").Get("index");
                material.extension.specularGlossinessTexture = &textures[index.Get<int>()];
                auto texCoordSet = ext->second.Get("specularGlossinessTexture").Get("texCoord");
                material.texCoordSets.specularGlossiness = texCoordSet.Get<int>();
                material.pbrWorkflows.specularGlossiness = true;
            }
            if (ext->second.Has("diffuseTexture")) {
                auto index = ext->second.Get("diffuseTexture").Get("index");
                material.extension.diffuseTexture = &textures[index.Get<int>()];
            }
            if (ext->second.Has("diffuseFactor")) {
                auto factor = ext->second.Get("diffuseFactor");
                for (uint32_t i = 0; i < factor.ArrayLen(); i++) {
                    auto val = factor.Get(i);
                    material.extension.diffuseFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                }
            }
            if (ext->second.Has("specularFactor")) {
                auto factor = ext->second.Get("specularFactor");
                for (uint32_t i = 0; i < factor.ArrayLen(); i++) {
                    auto val = factor.Get(i);
                    material.extension.specularFactor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                }
            }
        }
        materials.push_back(material);
    }
}

void GltfModel::loadFromFile(const moon::utils::PhysicalDevice& device, VkCommandBuffer commandBuffer)
{
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;

    if (std::string error{}, warning{}; isBinary(filename) ?
        gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.string().c_str()) :
        gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.string().c_str()))
    {
        loadTextures(device, commandBuffer, gltfModel);
        loadMaterials(gltfModel);

        for(auto& instance: instances){
            uint32_t indexStart = 0;
            for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
                loadNode(&instance, device, device.device(), nullptr, nodeIndex, gltfModel, indexStart);
            }
        }

        std::vector<uint32_t> indexBuffer;
        std::vector<Vertex> vertexBuffer;
        for (const auto& nodeIndex: gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0].nodes) {
            loadVertexBuffer(gltfModel.nodes[nodeIndex], gltfModel, indexBuffer, vertexBuffer);
        }
        calculateNodeTangent(vertexBuffer, indexBuffer);

        loadSkins(gltfModel);
        if (gltfModel.animations.size() > 0) {
            loadAnimations(gltfModel);
        }

        for(auto& instance : instances){
            for (auto& node : instance.nodes) {
                node->update();
            }
        }

        utils::createDeviceBuffer(device, device.device(), commandBuffer, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexCache, vertices);
        utils::createDeviceBuffer(device, device.device(), commandBuffer, indexBuffer.size() * sizeof(uint32_t), indexBuffer.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexCache, indices);
    }
}

void GltfModel::createDescriptorPool() {
    nodeDescriptorSetLayout = moon::interfaces::Model::createNodeDescriptorSetLayout(device);
    materialDescriptorSetLayout = moon::interfaces::Model::createMaterialDescriptorSetLayout(device);

    uint32_t nodesCount = std::accumulate(instances.begin(), instances.end(), 0, [](const uint32_t& count, const auto& instance) {
        return count + std::accumulate(instance.nodes.begin(), instance.nodes.end(), 0, [](const uint32_t& count, Node* node) {
            return count + node->meshCount();
        });
    });

    std::vector<const utils::vkDefault::DescriptorSetLayout*> materialDescriptorSetLayouts(materials.size(), &materialDescriptorSetLayout);
    std::vector<const utils::vkDefault::DescriptorSetLayout*> nodeDescriptorSetLayouts(nodesCount, &nodeDescriptorSetLayout);
    std::vector<const utils::vkDefault::DescriptorSetLayout*> descriptorSetLayouts;
    descriptorSetLayouts.insert(descriptorSetLayouts.end(), materialDescriptorSetLayouts.begin(), materialDescriptorSetLayouts.end());
    descriptorSetLayouts.insert(descriptorSetLayouts.end(), nodeDescriptorSetLayouts.begin(), nodeDescriptorSetLayouts.end());

    descriptorPool = utils::vkDefault::DescriptorPool(device, descriptorSetLayouts, 1);
}

void GltfModel::createDescriptorSet()
{
    for(auto& instance : instances){
        for (auto& node : instance.nodes){
            createNodeDescriptorSet(device, node , descriptorPool, nodeDescriptorSetLayout);
        }
    }

    for (auto &material : materials){
        createMaterialDescriptorSet(device, &material, descriptorPool, materialDescriptorSetLayout);
    }
}

void GltfModel::create(const moon::utils::PhysicalDevice& device, VkCommandPool commandPool)
{
    if(this->device == VK_NULL_HANDLE)
    {
        CHECK_M(VkPhysicalDevice(device) == VK_NULL_HANDLE, std::string("[ GltfModel::create ] VkPhysicalDevice is VK_NULL_HANDLE"));
        CHECK_M(VkDevice(device.device()) == VK_NULL_HANDLE, std::string("[ GltfModel::create ] VkDevice is VK_NULL_HANDLE"));
        CHECK_M(commandPool == VK_NULL_HANDLE, std::string("[ GltfModel::create ] VkCommandPool is VK_NULL_HANDLE"));

        this->device = device.device();

        VkCommandBuffer commandBuffer = moon::utils::singleCommandBuffer::create(device.device(),commandPool);
        loadFromFile(device, commandBuffer);
        moon::utils::singleCommandBuffer::submit(device.device(), device.device()(0,0), commandPool, &commandBuffer);
        destroyCache();

        createDescriptorPool();
        createDescriptorSet();
    }
}

void GltfModel::render(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets, uint32_t &primitiveCount, uint32_t pushConstantSize, uint32_t pushConstantOffset, void* pushConstant){
    for (auto node: instances[frameIndex].nodes){
        renderNode(node, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets, primitiveCount, pushConstantSize, pushConstantOffset, pushConstant);
    }
}

void GltfModel::renderBB(uint32_t frameIndex, VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetsCount, VkDescriptorSet* descriptorSets){
    for (auto node: instances[frameIndex].nodes){
        renderNodeBB(node, commandBuffer, pipelineLayout, descriptorSetsCount, descriptorSets);
    }
}

}
