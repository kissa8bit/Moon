#include "gltfmodel.h"

namespace moon::models {

namespace {

class Extractor {
public:
    Extractor(const tinygltf::Material& material, const utils::Textures& textures)
        : material(material), textures(textures)
    {
        if(!textures.empty()){
            emptyTexture = &textures.back();
        }
    }

    template<typename TextureInfo, typename FactorType = double>
    interfaces::Material::TextureParameters operator()(const TextureInfo& textureInfo, const std::vector<FactorType>& factor = {}) const {
        interfaces::Material::TextureParameters textureParameters(emptyTexture);

        if (textureInfo.index != -1) {
            textureParameters.texture = &textures.at(textureInfo.index);
            textureParameters.coordSet = textureInfo.texCoord;
        }

        if (factor.size() >= 4) {
            textureParameters.factor = math::Vector<float, 4>(factor[0], factor[1], factor[2], factor[3]);
        }

        return textureParameters;
    }

    interfaces::Material::TextureParameters operator()(const tinygltf::Value& extensions, const std::string& texName, const std::string& factorName = "") const {
        interfaces::Material::TextureParameters textureParameters(emptyTexture);

        if (extensions.Has(texName)) {
            if (const auto& texture = extensions.Get(texName); !(texture == nullValue)){
                getTexure(textureParameters, texture);
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
    void getTexure(interfaces::Material::TextureParameters& textureParameters, const tinygltf::Value& texture) const {
        const auto& index = texture.Get("index");
        if (index == nullValue) return;
        textureParameters.texture = &textures[index.Get<int>()];

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

    const tinygltf::Material& material;
    const utils::Textures& textures;
    const utils::Texture* emptyTexture{nullptr};

    inline static const tinygltf::Value nullValue{};
};

}

void GltfModel::loadMaterials(const tinygltf::Model& gltfModel) {
    static const std::unordered_map<std::string, interfaces::Material::AlphaMode> alphaModeMap = {
        {"OPAQUE", interfaces::Material::AlphaMode::ALPHAMODE_OPAQUE},
        {"MASK", interfaces::Material::AlphaMode::ALPHAMODE_MASK},
        {"BLEND", interfaces::Material::AlphaMode::ALPHAMODE_BLEND}
    };

    for (const tinygltf::Material& mat : gltfModel.materials)
    {
        Extractor extractor(mat, textures);
        const auto& pbr = mat.pbrMetallicRoughness;

        auto& material = materials.emplace_back();
        material.baseColor = extractor(pbr.baseColorTexture, pbr.baseColorFactor);
        material.metallicRoughness = extractor(pbr.metallicRoughnessTexture, {pbr.metallicFactor, pbr.roughnessFactor});
        material.normal = extractor(mat.normalTexture);
        material.emissive = extractor(mat.emissiveTexture, mat.emissiveFactor);
        material.occlusion = extractor(mat.occlusionTexture);
        material.alphaMode = alphaModeMap.at(mat.alphaMode);
        material.alphaCutoff = mat.alphaCutoff;

        if (auto extIt = mat.extensions.find("KHR_materials_pbrSpecularGlossiness"); extIt != mat.extensions.end()) {
            const auto& extensions = extIt->second;

            if (extensions.Has("specularGlossinessTexture")) {
                material.pbrWorkflows = interfaces::Material::PbrWorkflow::SPECULAR_GLOSSINESS;
            }

            material.extensions.specularGlossiness = extractor(extensions, "specularGlossinessTexture", "specularFactor");
            material.extensions.diffuse = extractor(extensions, "diffuseTexture", "diffuseFactor");
        }
    }
}

}