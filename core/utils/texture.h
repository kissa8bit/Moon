#ifndef MOON_UTILS_TEXTURE_H
#define MOON_UTILS_TEXTURE_H

#include <filesystem>
#include <unordered_map>
#include <optional>

#include <vulkan.h>
#include "buffer.h"
#include "vkdefault.h"
#include "device.h"

namespace moon::utils {

struct TextureSampler {
    VkFilter magFilter{ VK_FILTER_LINEAR };
    VkFilter minFilter{ VK_FILTER_LINEAR };
    VkSamplerAddressMode addressModeU{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
    VkSamplerAddressMode addressModeV{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
    VkSamplerAddressMode addressModeW{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
};

struct TextureImage {
    utils::vkDefault::Image image;
    utils::vkDefault::ImageView imageView;
    utils::vkDefault::Sampler sampler;

    int width{ -1 };
    int height{ -1 };
    int channels{ -1 };
    VkDeviceSize size{ 0 };
    float mipLevel{ 0.0f };
    uint32_t mipLevels{ 1 };
    VkFormat format{ VK_FORMAT_R8G8B8A8_SRGB };

    Buffer cache;

    ~TextureImage();
    TextureImage() = default;
    TextureImage(const TextureImage&) = delete;
    TextureImage& operator=(const TextureImage&) = delete;
    TextureImage(TextureImage&&) noexcept;
    TextureImage& operator=(TextureImage&&) noexcept;
    void swap(TextureImage& other) noexcept;

    void makeCache(
            VkPhysicalDevice            physicalDevice,
            VkDevice                    device,
            const std::vector<void*>&   buffers);

    VkResult create(
            VkPhysicalDevice            physicalDevice,
            VkDevice                    device,
            VkCommandBuffer             commandBuffer,
            VkImageCreateFlags          flags,
            const uint32_t&             imageCount,
		    VkFormat                    format,
            const TextureSampler&       textureSampler,
            std::optional<uint32_t>     mipLevelsOpt = std::nullopt);
};

class Texture{
protected:
    utils::vkDefault::Paths paths;
    TextureImage image;

    Texture(const utils::vkDefault::Paths& paths);

public:
    virtual ~Texture() = default;
    Texture() = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;
    void swap(Texture& other) noexcept;

    void destroyCache();

    Texture(VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkCommandBuffer                 commandBuffer,
            int                             width,
            int                             height,
            void*                           buffer,
		    VkFormat                        format = VK_FORMAT_R8G8B8A8_SRGB,
            const TextureSampler&           textureSampler = TextureSampler{},
            std::optional<uint32_t>         mipLevels = std::nullopt);

#ifdef USE_STB_IMAGE
    Texture(const std::filesystem::path&    path,
            VkPhysicalDevice                physicalDevice,
            VkDevice                        device,
            VkCommandBuffer                 commandBuffer,
            VkFormat                        format = VK_FORMAT_R8G8B8A8_SRGB,
            const TextureSampler&           textureSampler = TextureSampler{},
            std::optional<uint32_t>         mipLevels = std::nullopt);
#endif

    VkImageView imageView() const;
    VkSampler sampler() const;
    VkDescriptorImageInfo descriptorImageInfo() const;

    enum class EmptyType {
        Black,
        White
	};

    static Texture createEmpty(const PhysicalDevice&, VkCommandPool, EmptyType type = EmptyType::Black);
    static Texture createEmpty(const PhysicalDevice&, VkCommandBuffer, EmptyType type = EmptyType::Black);
};

using Textures = std::unordered_map<int, Texture>;
using TextureMap = std::unordered_map<std::string, Texture>;

class CubeTexture: public Texture {
public:
    CubeTexture() = default;
    ~CubeTexture() = default;
    CubeTexture(CubeTexture&&) noexcept = default;
    CubeTexture& operator=(CubeTexture&&) noexcept = default;
    CubeTexture(Texture&& texture);
#ifdef USE_STB_IMAGE
    CubeTexture(const utils::vkDefault::Paths&      path,
                VkPhysicalDevice                    physicalDevice,
                VkDevice                            device,
                VkCommandBuffer                     commandBuffer,
                VkFormat                            format = VK_FORMAT_R8G8B8A8_SRGB,
                const TextureSampler&               textureSampler = TextureSampler{},
                std::optional<uint32_t>             mipLevels = std::nullopt);
#endif
};

}
#endif // MOON_UTILS_TEXTURE_H
