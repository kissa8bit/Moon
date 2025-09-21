#ifndef MOON_UTILS_ATTACHMENTS_H
#define MOON_UTILS_ATTACHMENTS_H

#include <vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>

#include "vkdefault.h"
#include "texture.h"

namespace moon::utils {

struct Attachment {
    utils::vkDefault::Image image;
    utils::vkDefault::ImageView imageView;
    utils::vkDefault::ImageInfo imageInfo;
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};

    Attachment() = default;
    Attachment(const Attachment & other) = delete;
    Attachment& operator=(const Attachment& other) = delete;
    Attachment(Attachment&& other) noexcept;
    Attachment& operator=(Attachment&& other) noexcept;
    void swap(Attachment& other) noexcept;

    Attachment(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo& imageInfo, VkImageUsageFlags usage);

    void transitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout);
    void copyFrom(VkCommandBuffer commandBuffer, Attachment& dst);
    void copyTo(VkCommandBuffer commandBuffer, Attachment& dst);
    void copyFrom(VkCommandBuffer commandBuffer, VkBuffer bfr);
    void copyTo(VkCommandBuffer commandBuffer, VkBuffer bfr);
    void clear(VkCommandBuffer commandBuffer, VkClearColorValue clearColorValue = { 0, 0, 0, 0 });
    void blitDown(VkCommandBuffer commandBuffer, Attachment& dst, float blitFactor);
    void blitUp(VkCommandBuffer commandBuffer, Attachment& dst, float blitFactor);
    void downscale(VkCommandBuffer commandBuffer, Attachment& intermBfr, size_t count, float factor);
    void createMipLevels(VkCommandBuffer commandBuffer);

    static VkAttachmentDescription imageDescription(VkFormat format);
    static VkAttachmentDescription imageDescription(VkFormat format, VkImageLayout layout);
};

class Attachments {
private:
    std::vector<Attachment> instances;
    utils::vkDefault::Sampler imageSampler;
    utils::vkDefault::ImageInfo imageInfo;
    VkClearValue imageClearValue{};

public:
    Attachments() = default;
    Attachments(const Attachments& other) = delete;
    Attachments& operator=(const Attachments& other) = delete;
    Attachments(Attachments&& other) noexcept;
    Attachments& operator=(Attachments&& other) noexcept;
    void swap(Attachments& other) noexcept;

    Attachments(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo& imageInfo, VkImageUsageFlags usage, const VkClearValue& clear = {{0.0f, 0.0f, 0.0f, 0.0f}}, VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO });

    std::vector<VkImage> getImages() const;

    const VkImage& image(size_t i) const { return instances[i].image; }
    const VkImageView& imageView(size_t i) const { return instances[i].imageView; }
    const Attachment& attachment(size_t i) const { return instances[i]; }
    const VkSampler& sampler() const {return imageSampler;}
    const VkFormat& format() const { return imageInfo.Format; }
    const uint32_t& count() const { return imageInfo.Count; }
    const VkClearValue& clearValue() const { return imageClearValue; }
    VkDescriptorImageInfo descriptorImageInfo(size_t i) const {
        VkDescriptorImageInfo imageInfo;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = imageView(i);
        imageInfo.sampler = sampler();
        return imageInfo;
    }

    static VkAttachmentDescription imageDescription(VkFormat format);
    static VkAttachmentDescription imageDescription(VkFormat format, VkImageLayout layout);
    static VkAttachmentDescription depthStencilDescription(VkFormat format);
    static VkAttachmentDescription depthDescription(VkFormat format);
};

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo image, uint32_t attachmentsCount, Attachments* pAttachments, VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VkSamplerCreateInfo samplerInfo = VkSamplerCreateInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO });

struct AttachmentsDatabase {
    struct data{
        bool enable{false};
        const Attachments* pImages{nullptr};
    };

    std::string defaultEmptyTexture;
    std::unordered_map<std::string, Texture*> emptyTexturesMap;
    std::unordered_map<std::string, data> attachmentsMap;

    AttachmentsDatabase() = default;
    AttachmentsDatabase(const std::string& emptyTextureId, Texture* emptyTexture);
    AttachmentsDatabase(const AttachmentsDatabase&) = default;
    AttachmentsDatabase& operator=(const AttachmentsDatabase&) = default;

    void destroy();

    bool addEmptyTexture(const std::string& id, Texture* emptyTexture);
    bool addAttachmentData(const std::string& id, bool enable, const Attachments* pImages);
    bool enable(const std::string& id) const;
    const Attachments* get(const std::string& id) const;
    const Texture* getEmpty(const std::string& id = "") const;
    VkImageView imageView(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkSampler sampler(const std::string& id, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkDescriptorImageInfo descriptorImageInfo(const std::string& id, const uint32_t imageIndex, const std::optional<std::string>& emptyTextureId = std::nullopt) const;
    VkDescriptorImageInfo descriptorEmptyInfo(const std::string& id = "") const;
};

}
#endif // MOON_UTILS_ATTACHMENTS_H
