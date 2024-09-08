#include "gltfmodel.h"

namespace moon::models {

void GltfModel::loadMaterials(const tinygltf::Model& gltfModel) {
    const utils::Texture* emptyTexture = &textures.back();
    for (const tinygltf::Material& mat : gltfModel.materials)
    {
        interfaces::Material material(emptyTexture);
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
            material.baseColorFactor = math::Vector<float, 4>(factor[0], factor[1], factor[2], factor[3]);
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
                material.alphaMode = interfaces::Material::ALPHAMODE_BLEND;
            }
            if (param.string_value == "MASK") {
                material.alphaCutoff = 0.5f;
                material.alphaMode = interfaces::Material::ALPHAMODE_MASK;
            }
        }
        if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end()) {
            const auto& alphaCutoff = mat.additionalValues.at("alphaCutoff");
            material.alphaCutoff = static_cast<float>(alphaCutoff.Factor());
        }
        if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end()) {
            const auto& factor = mat.additionalValues.at("emissiveFactor").ColorFactor();
            material.emissiveFactor = math::Vector<float, 4>(factor[0], factor[1], factor[2], 1.0f);
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

}