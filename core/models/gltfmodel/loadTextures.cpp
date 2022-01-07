#include "gltfmodel.h"
#include "gltfutils.h"

namespace moon::models {

void GltfModel::loadTextures(const tinygltf::Model& gltfModel, const utils::PhysicalDevice& device, VkCommandBuffer commandBuffer) {
    for (const tinygltf::Texture& tex : gltfModel.textures) {
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
        textures.emplace_back(device, device.device(), commandBuffer, width, height, buffer.data(), textureSampler);
    }
    textures.push_back(utils::Texture::empty(device, commandBuffer));
}

} // moon::models