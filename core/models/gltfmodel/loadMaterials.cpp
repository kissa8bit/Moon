#include "gltfmodel.h"
#include "gltfutils.h"

#include <math/linearAlgebra.h>

namespace moon::models {

namespace {

utils::Texture loadTexture(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer, VkFormat format, int texIndex) {
    const tinygltf::Texture& tex = gltfModel.textures[texIndex];
    const tinygltf::Image& gltfimage = gltfModel.images[tex.source];

    const uint32_t downsampleWidth = 1, downsampleHeight = 1;

    const auto width = gltfimage.width / downsampleWidth;
    const auto height = gltfimage.height / downsampleHeight;

    std::vector<uint8_t> buffer(4 * width * height);

    for (uint32_t i = 0, offset = 0; i < height; ++i) {
        for (uint32_t j = 0; j < width; ++j) {
            uint32_t line = gltfimage.component * (gltfimage.width * downsampleHeight * i + downsampleHeight * j);

            buffer[offset + 3] = 255;
            for (uint32_t k = 0; k < gltfimage.component; ++k) {
                buffer[offset + k] = gltfimage.image[line + k];
            }
            offset += 4;
        }
    }

    utils::TextureSampler textureSampler{};
    if (isValid(tex.sampler)) {
        const auto& samplers = gltfModel.samplers.at(tex.sampler);
        textureSampler.minFilter = getVkFilterMode(samplers.minFilter);
        textureSampler.magFilter = getVkFilterMode(samplers.magFilter);
        textureSampler.addressModeV = textureSampler.addressModeW = getVkWrapMode(samplers.wrapT);
        textureSampler.addressModeU = getVkWrapMode(samplers.wrapS);
    }
    return utils::Texture(device, device.device(), commandBuffer, width, height, buffer.data(), format, textureSampler);
}

class Extractor {
public:
    const utils::Texture* getEmptyTexture() {
        return textures.empty() ? nullptr : &textures[-1];
    }

    Extractor(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer, utils::Textures& textures)
        : gltfModel(gltfModel), device(device), commandBuffer(commandBuffer), textures(textures), emptyTexture(getEmptyTexture())
    {}

    template<typename TextureInfo, typename FactorType = double>
    interfaces::Material::TextureParameters operator()(const TextureInfo& textureInfo, VkFormat format, const std::vector<FactorType>& factor = {}) const {
        interfaces::Material::TextureParameters textureParameters(emptyTexture);

        if (const auto index = textureInfo.index; isValid(index)) {
            if (textures.find(index) == textures.end()) {
                textures[index] = loadTexture(gltfModel, device, commandBuffer, format, index);
            }
            textureParameters.texture = &textures[index];
            textureParameters.coordSet = textureInfo.texCoord;
        }

        if (factor.size() >= 4) {
            textureParameters.factor = math::vec4(factor[0], factor[1], factor[2], factor[3]);
        }

        return textureParameters;
    }

    interfaces::Material::TextureParameters operator()(const tinygltf::Value& extensions, const std::string& texName, VkFormat format, const std::string& factorName = "") const {
        interfaces::Material::TextureParameters textureParameters(emptyTexture);

        if (extensions.Has(texName)) {
            if (const auto& texture = extensions.Get(texName); !(texture == nullValue)){
                getTexure(textureParameters, texture, format);
            }
        }

        if (factorName != "" && extensions.Has(factorName)) {
            if (const auto& factor = extensions.Get(factorName); !(factor == nullValue)) {
                getFactor(textureParameters, factor);
            }
        }

        return textureParameters;
    }

private:
    void getTexure(interfaces::Material::TextureParameters& textureParameters, const tinygltf::Value& texture, VkFormat format) const {
        const auto& index = texture.Get("index");
        if (index == nullValue) return;
		const int indexValue = index.Get<int>();
        if (textures.find(indexValue) == textures.end()) {
            textures[indexValue] = loadTexture(gltfModel, device, commandBuffer, format, indexValue);
        }
        textureParameters.texture = &textures[indexValue];

        const auto& texCoordSet = texture.Get("texCoord");
        if (texture == nullValue) return;
        textureParameters.coordSet = texCoordSet.Get<int>();
    };

    void getFactor(interfaces::Material::TextureParameters& textureParameters,  const tinygltf::Value& factor) const {
        for (uint32_t i = 0; i < factor.ArrayLen(); i++) {
            const auto& val = factor.Get(i);
            textureParameters.factor[i] = val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
        }
    };

    const tinygltf::Model& gltfModel;
    const utils::PhysicalDevice& device;
    VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
    utils::Textures& textures;
    const utils::Texture* emptyTexture{nullptr};

    inline static const tinygltf::Value nullValue{};
};

}

void GltfModel::loadMaterials(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
    static const std::unordered_map<std::string, interfaces::Material::AlphaMode> alphaModeMap = {
        {"OPAQUE", interfaces::Material::AlphaMode::ALPHAMODE_OPAQUE},
        {"MASK", interfaces::Material::AlphaMode::ALPHAMODE_MASK},
        {"BLEND", interfaces::Material::AlphaMode::ALPHAMODE_BLEND}
    };

    textures[-1] = utils::Texture::createEmpty(device, commandBuffer);

    Extractor extractor(gltfModel, device, commandBuffer, textures);
    for (const tinygltf::Material& mat : gltfModel.materials)
    {
        const auto& pbr = mat.pbrMetallicRoughness;

        auto& material = materials.emplace_back();
        material.baseColor = extractor(pbr.baseColorTexture, VK_FORMAT_R8G8B8A8_SRGB, pbr.baseColorFactor);
        material.metallicRoughness = extractor(pbr.metallicRoughnessTexture, VK_FORMAT_R8G8B8A8_UNORM);
        material.metallicRoughness.factor[interfaces::Material::metallicIndex] = pbr.metallicFactor;
        material.metallicRoughness.factor[interfaces::Material::roughnessIndex] = pbr.roughnessFactor;
        material.normal = extractor(mat.normalTexture, VK_FORMAT_R8G8B8A8_UNORM);
        material.emissive = extractor(mat.emissiveTexture, VK_FORMAT_R8G8B8A8_SRGB, mat.emissiveFactor);
        material.occlusion = extractor(mat.occlusionTexture, VK_FORMAT_R8G8B8A8_UNORM);
        material.alphaMode = alphaModeMap.at(mat.alphaMode);
        material.alphaCutoff = mat.alphaCutoff;

        if (auto extIt = mat.extensions.find("KHR_materials_pbrSpecularGlossiness"); extIt != mat.extensions.end()) {
            const auto& extensions = extIt->second;

            if (extensions.Has("specularGlossinessTexture")) {
                material.pbrWorkflows = interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS;
            }

            material.extensions.specularGlossiness = extractor(extensions, "specularGlossinessTexture", VK_FORMAT_R8G8B8A8_UNORM, "specularFactor");
            material.extensions.diffuse = extractor(extensions, "diffuseTexture", VK_FORMAT_R8G8B8A8_SRGB, "diffuseFactor");
        }
    }
    auto& emptyMaterial = materials.emplace_back(extractor.getEmptyTexture());
}

} // moon::models