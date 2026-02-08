#include "texture.h"

#include "operations.h"
#include "memory.h"

#ifdef USE_STB_IMAGE
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif
#endif

#include <cmath>
#include <cstring>

namespace moon::utils {

void TextureImage::swap(TextureImage& other) noexcept {
    std::swap(image, other.image);
    std::swap(imageView, other.imageView);
    std::swap(sampler, other.sampler);
    std::swap(format, other.format);
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(channels, other.channels);
    std::swap(size, other.size);
    std::swap(cache, other.cache);
    std::swap(mipLevel, other.mipLevel);
    std::swap(mipLevels, other.mipLevels);
}

TextureImage::TextureImage(TextureImage&& other) noexcept {
    swap(other);
}

TextureImage& TextureImage::operator=(TextureImage&& other) noexcept {
    swap(other);
    return *this;
}

TextureImage::~TextureImage() {}

void TextureImage::makeCache(
        VkPhysicalDevice            physicalDevice,
        VkDevice                    device,
        const std::vector<void*>&   buffers)
{
    if(width == -1 || height == -1 || channels == -1) throw std::runtime_error("[TextureImage::makeCache] : texture sizes not init");

    cache = utils::vkDefault::Buffer(physicalDevice, device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    Memory::instance().nameMemory(cache, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", TextureImage::makeCache, cache");

    for (uint32_t i = 0, ds = 4 * width * height; i < buffers.size(); i++) {
        cache.copy(buffers[i], i * ds, ds);
    }
}

VkResult TextureImage::create(
        VkPhysicalDevice            physicalDevice,
        VkDevice                    device,
        VkCommandBuffer             commandBuffer,
        VkImageCreateFlags          flags,
        const uint32_t&             imageCount,
        VkFormat                    form,
        const TextureSampler&       textureSampler,
        std::optional<uint32_t>     mipLevelsOpt)
{
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("[TextureImage::create] : invalid texture size (width/height must be > 0)");
    }

    format = form;
    mipLevels = mipLevelsOpt.value_or(static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(std::max(width, height))))) + 1);
    image = utils::vkDefault::Image(physicalDevice,
                                    device,
                                    flags,
                                    {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
                                    imageCount,
                                    mipLevels,
                                    VK_SAMPLE_COUNT_1_BIT,
                                    format,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (mipLevels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Memory::instance().nameMemory(image, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", TextureImage::create, image");

    texture::transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels, 0, imageCount);
    texture::copy(commandBuffer, cache, image, {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}, imageCount);

    if(mipLevels == 1){
        texture::transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels, 0, imageCount);
    } else {
        texture::generateMipmaps(physicalDevice, commandBuffer, image, format, width, height, mipLevels, 0, imageCount);
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = textureSampler.magFilter;
    samplerInfo.minFilter = textureSampler.minFilter;
    samplerInfo.addressModeU = textureSampler.addressModeU;
    samplerInfo.addressModeV = textureSampler.addressModeV;
    samplerInfo.addressModeW = textureSampler.addressModeW;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.mipLodBias = 0.0f;

    const VkImageViewType type = (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    imageView = utils::vkDefault::ImageView(device, image, type, format, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels, 0, imageCount);
    sampler = utils::vkDefault::Sampler(device, samplerInfo);
    return VK_SUCCESS;
}

Texture::Texture(const utils::vkDefault::Paths& path) : paths(path) {}

void Texture::swap(Texture& other) noexcept {
    std::swap(paths, other.paths);
    std::swap(image, other.image);
}

Texture::Texture(Texture&& other) noexcept {
    swap(other);
}

Texture& Texture::operator=(Texture&& other) noexcept {
    swap(other);
    return *this;
}

void Texture::destroyCache(){
    image.cache = Buffer();
}

Texture::Texture(
        VkPhysicalDevice        physicalDevice,
        VkDevice                device,
        VkCommandBuffer         commandBuffer,
        int                     width,
        int                     height,
        void*                   buffer,
        VkFormat                format,
        const TextureSampler&   textureSampler,
        std::optional<uint32_t> mipLevels)
{
    image.width = width;
    image.height = height;
    image.channels = 4;
    image.size = 4 * image.width * image.height;
    image.makeCache(physicalDevice, device, { buffer });
    CHECK(image.create(physicalDevice, device, commandBuffer, 0, 1, format, textureSampler, mipLevels));
}

#ifdef USE_STB_IMAGE
Texture::Texture(
        const std::filesystem::path&    path,
        VkPhysicalDevice                physicalDevice,
        VkDevice                        device,
        VkCommandBuffer                 commandBuffer,
        VkFormat                        format,
        const TextureSampler&           textureSampler,
        std::optional<uint32_t>         mipLevels) : paths({ path })
{
    if(paths.empty()) throw std::runtime_error("[Texture::create] : no paths to texture");

    stbi_uc* buffer = stbi_load(paths.front().string().c_str(), &image.width, &image.height, &image.channels, STBI_rgb_alpha);
    image.size = 4 * image.width * image.height;
    if(!buffer) throw std::runtime_error("[Texture::create] : failed to load texture image!");
    image.makeCache(physicalDevice, device, { buffer });
    stbi_image_free(buffer);

    CHECK(image.create(physicalDevice, device, commandBuffer, 0, 1, format, textureSampler, mipLevels));
}
#endif

VkImageView Texture::imageView() const {return image.imageView;}
VkSampler Texture::sampler() const {return image.sampler;}

VkDescriptorImageInfo Texture::descriptorImageInfo() const {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = imageView();
    imageInfo.sampler = sampler();
    return imageInfo;
}

CubeTexture::CubeTexture(Texture&& texture) : Texture(std::move(texture)){}

#ifdef USE_STB_IMAGE
CubeTexture::CubeTexture(
        const utils::vkDefault::Paths&  path, 
        VkPhysicalDevice                physicalDevice, 
        VkDevice                        device,
        VkCommandBuffer                 commandBuffer,
        VkFormat                        format,
        const TextureSampler&           textureSampler,
        std::optional<uint32_t>         mipLevels ) : Texture(path)
{
    if (paths.size() != 6) throw std::runtime_error("[CubeTexture::create] : must be 6 images");

    int maxWidth = -1, maxHeight = -1, maxChannels = -1;
    std::vector<void*> buffers;
    for(uint32_t i = 0; i < 6; i++) {
        buffers.push_back(stbi_load(paths[i].string().c_str(), &image.width, &image.height, &image.channels, STBI_rgb_alpha));
        image.size += 4 * image.width * image.height;
        if (!buffers.back()) throw std::runtime_error("[CubeTexture::create] : failed to load texture image!");

        if (maxWidth == -1 && maxHeight == -1 && maxChannels == -1) {
            maxWidth = image.width; maxHeight = image.height; maxChannels = image.channels;
        }
        else if (maxWidth != image.width && maxHeight != image.height && maxChannels != image.channels) {
            throw std::runtime_error("[CubeTexture::create] : images must be same size!");
        }
    }

    image.makeCache(physicalDevice, device, buffers);
    for(auto& buffer : buffers) stbi_image_free(buffer);

    CHECK(image.create(physicalDevice, device, commandBuffer, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 6, format, textureSampler, mipLevels));
}
#endif

Texture Texture::createEmpty(const PhysicalDevice& device, VkCommandPool commandPool, Texture::EmptyType type){
    VkCommandBuffer commandBuffer = singleCommandBuffer::create(device.device(),commandPool);
    Texture tex = Texture::createEmpty(device, commandBuffer, type);
    singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer);
    tex.destroyCache();
    return tex;
};

Texture Texture::createEmpty(const PhysicalDevice& device, VkCommandBuffer commandBuffer, Texture::EmptyType type) {
    uint32_t buffer = 0;
    switch (type) {
        case Texture::EmptyType::Black:
            buffer = 0xff000000; // RGBA for black
            break;
        case Texture::EmptyType::White:
            buffer = 0xffffffff; // RGBA for white
            break;
	}
    const int width = 1, height = 1;
    return Texture(device, device.device(), commandBuffer, width, height, &buffer);
}

}
