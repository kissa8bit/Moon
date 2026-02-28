#include "attachments.h"

#include "operations.h"
#include "vkdefault.h"
#include "memory.h"
#include "texture.h"

#include <algorithm>
#include <iterator>

namespace moon::utils {

Attachment::Attachment(Attachment&& other) noexcept {
    swap(other);
};

Attachment& Attachment::operator=(Attachment&& other) noexcept {
    swap(other);
    return *this;
};

void Attachment::swap(Attachment& other) noexcept {
    std::swap(image, other.image);
    std::swap(imageView, other.imageView);
    std::swap(imageInfo, other.imageInfo);
    std::swap(layout, other.layout);
}

Attachment::Attachment(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo& imageInfo, VkImageUsageFlags usage)
    : imageInfo(imageInfo)
{
    const auto depthFormats = image::depthFormats();
    const bool isDepth = std::any_of(depthFormats.begin(), depthFormats.end(), [&imageInfo](const VkFormat& format) {return imageInfo.Format == format; });
    const VkImageAspectFlagBits imageAspectFlagBits = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    image = utils::vkDefault::Image(physicalDevice, device, 0, { imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1, 1, VK_SAMPLE_COUNT_1_BIT, imageInfo.Format, VK_IMAGE_LAYOUT_UNDEFINED, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    imageView = utils::vkDefault::ImageView(device, image, VK_IMAGE_VIEW_TYPE_2D, imageInfo.Format, imageAspectFlagBits, 1, 0, 1);
}


void Attachment::transitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout)
{
    if (layout != newLayout)
    {
        texture::transitionLayout(commandBuffer, image, layout, newLayout, 1, 0, 1);
        layout = newLayout;
    }
}

void Attachment::copyFrom(VkCommandBuffer commandBuffer, Attachment& dst)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    dst.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    utils::texture::copy(commandBuffer, dst.image, image, VkExtent3D{ imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1);
}

void Attachment::copyTo(VkCommandBuffer commandBuffer, Attachment& dst)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    dst.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    utils::texture::copy(commandBuffer, image, dst.image, VkExtent3D{ imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1);
}

void Attachment::copyFrom(VkCommandBuffer commandBuffer, VkBuffer bfr)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    utils::texture::copy(commandBuffer, bfr, image, { imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1);
}

void Attachment::copyTo(VkCommandBuffer commandBuffer, VkBuffer bfr)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    texture::copy(commandBuffer, image, bfr, VkExtent3D{ imageInfo.Extent.width, imageInfo.Extent.height, 1 }, 1);
}

void Attachment::clear(VkCommandBuffer commandBuffer, VkClearColorValue clearColorValue)
{
    VkImageSubresourceRange ImageSubresourceRange{};
    const auto depthFormats = image::depthFormats();
    const bool isDepth = std::any_of(depthFormats.begin(), depthFormats.end(), [this](const VkFormat& format) { return imageInfo.Format == format; });
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (isDepth) {
        if (imageInfo.Format == VK_FORMAT_D32_SFLOAT_S8_UINT || imageInfo.Format == VK_FORMAT_D24_UNORM_S8_UINT) {
            aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        } else {
            aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        }
    }

    ImageSubresourceRange.aspectMask = aspect;
    ImageSubresourceRange.baseMipLevel = 0;
    ImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    ImageSubresourceRange.baseArrayLayer = 0;
    ImageSubresourceRange.layerCount = 1;

    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(commandBuffer, image, layout, &clearColorValue, 1, &ImageSubresourceRange);
}

void Attachment::blitDown(VkCommandBuffer commandBuffer, Attachment& dst, float blitFactor)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    dst.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    dst.clear(commandBuffer);
    utils::texture::blitDown(commandBuffer, image, 0, dst.image, 0, imageInfo.Extent.width, imageInfo.Extent.height, 0, 1, blitFactor);
}

void Attachment::blitUp(VkCommandBuffer commandBuffer, Attachment& dst, float blitFactor)
{
    transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    dst.transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    dst.clear(commandBuffer);
    utils::texture::blitUp(commandBuffer, image, 0, dst.image, 0, imageInfo.Extent.width, imageInfo.Extent.height, 0, 1, blitFactor);
}

void Attachment::downscale(VkCommandBuffer commandBuffer, Attachment& intermBfr, size_t count, float factor)
{
    for (size_t i = count; i > 0; i--)
    {
        blitDown(commandBuffer, intermBfr, factor);
        intermBfr.blitUp(commandBuffer, *this, factor);
    }
}

Attachments::Attachments(Attachments&& other) noexcept {
    swap(other);
}

Attachments& Attachments::operator=(Attachments&& other) noexcept {
    swap(other);
    return *this;
}

void Attachments::swap(Attachments& other) noexcept {
    std::swap(instances, other.instances);
    std::swap(imageSampler, other.imageSampler);
    std::swap(imageInfo, other.imageInfo);
    std::swap(imageClearValue, other.imageClearValue);
}

Attachments::Attachments(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo& imageInfo, VkImageUsageFlags usage, const VkClearValue& clear, VkSamplerCreateInfo samplerInfo)
    : imageInfo(imageInfo), imageClearValue(clear)
{
    instances.resize(imageInfo.Count);
    for(auto& instance : instances){
        instance = Attachment(physicalDevice, device, imageInfo, usage);
        Memory::instance().nameMemory(instance.image, std::string(__FILE__) + " in line " + std::to_string(__LINE__) + ", attachments::create, instance " + std::to_string(&instance - &instances[0]));
    }
    imageSampler = utils::vkDefault::Sampler(device, samplerInfo);
}

VkAttachmentDescription Attachments::imageDescription(VkFormat format)
{
    VkAttachmentDescription description{};
        description.format = format;
        description.samples = VK_SAMPLE_COUNT_1_BIT;
        description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

VkAttachmentDescription Attachments::imageDescription(VkFormat format, VkImageLayout layout)
{
    VkAttachmentDescription description{};
        description.format = format;
        description.samples = VK_SAMPLE_COUNT_1_BIT;
        description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        description.finalLayout = layout;
    return description;
}

VkAttachmentDescription Attachments::depthDescription(VkFormat format)
{
    VkAttachmentDescription description{};
        description.format = format;
        description.samples = VK_SAMPLE_COUNT_1_BIT;
        description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

VkAttachmentDescription Attachments::depthStencilDescription(VkFormat format)
{
    VkAttachmentDescription description{};
        description.format = format;
        description.samples = VK_SAMPLE_COUNT_1_BIT;
        description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return description;
}

std::vector<VkImage> Attachments::getImages() const {
    std::vector<VkImage> images;
    for (const auto& instance: instances){
        images.push_back(instance.image);
    }
    return images;
}

void createAttachments(VkPhysicalDevice physicalDevice, VkDevice device, const utils::vkDefault::ImageInfo& imageInfo, uint32_t attachmentsCount, Attachments* pAttachments, VkImageUsageFlags usage, const VkClearValue& clear, VkSamplerCreateInfo samplerInfo){
    for(uint32_t i = 0; i < attachmentsCount; i++){
        pAttachments[i] = Attachments(physicalDevice, device, imageInfo, usage, clear, samplerInfo);
    }
}

AttachmentsDatabase::AttachmentsDatabase(const utils::ImageName& emptyTextureId, Texture* emptyTexture)
{
    defaultEmptyTexture = emptyTextureId;
    emptyTexturesMap[emptyTextureId] = emptyTexture;
}

void AttachmentsDatabase::destroy(){
    attachmentsMap.clear();
    emptyTexturesMap.clear();
    defaultEmptyTexture.clear();
}

bool AttachmentsDatabase::addEmptyTexture(const utils::ImageName& id, Texture* emptyTexture){
    if(emptyTexturesMap.count(id) > 0) return false;
    if(defaultEmptyTexture.empty()) defaultEmptyTexture = id;

    emptyTexturesMap[id] = emptyTexture;
    return true;
}

bool AttachmentsDatabase::addAttachmentData(const utils::AttachmentName& id, bool enable, const Attachments* pImages){
    if(attachmentsMap.count(id) > 0) return false;

    attachmentsMap[id] = data{enable, pImages};
    return true;
}

bool AttachmentsDatabase::enable(const utils::AttachmentName& id) const {
    auto it = attachmentsMap.find(id);
    return it != attachmentsMap.end() && it->second.enable;
}

const Attachments* AttachmentsDatabase::get(const utils::AttachmentName& id) const{
    auto it = attachmentsMap.find(id);
    return (it != attachmentsMap.end() && it->second.enable) ? it->second.pImages : nullptr;
}

const Texture* AttachmentsDatabase::getEmpty(const utils::ImageName& id) const {
    const auto texid = id.empty() ? defaultEmptyTexture : id;
    auto it = emptyTexturesMap.find(texid);
    return it != emptyTexturesMap.end() ? it->second : nullptr;
}

VkImageView AttachmentsDatabase::imageView(const utils::AttachmentName& id, const uint32_t imageIndex, const std::optional<utils::ImageName>& emptyTextureId) const {
    const Texture* emptyTexture = getEmpty(emptyTextureId ? *emptyTextureId : utils::ImageName());
    const auto attachment = get(id);
    if (attachment) {
        return attachment->imageView(imageIndex);
    }
    if (emptyTexture) {
        return emptyTexture->imageView();
    }
    return VK_NULL_HANDLE;
}

VkSampler AttachmentsDatabase::sampler(const utils::AttachmentName& id, const std::optional<utils::ImageName>& emptyTextureId) const {
    const Texture* emptyTexture = getEmpty(emptyTextureId ? *emptyTextureId : utils::ImageName());
    const auto attachment = get(id);
    if (attachment) {
        return attachment->sampler();
    }
    if (emptyTexture) {
        return emptyTexture->sampler();
    }
    return VK_NULL_HANDLE;
}

VkDescriptorImageInfo AttachmentsDatabase::descriptorImageInfo(const utils::AttachmentName& id, const uint32_t imageIndex, const std::optional<utils::ImageName>& emptyTextureId) const{
    VkDescriptorImageInfo res{};
    res.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    res.imageView = imageView(id, imageIndex, emptyTextureId);
    res.sampler = sampler(id, emptyTextureId);
    return res;
}

VkDescriptorImageInfo AttachmentsDatabase::descriptorEmptyInfo(const utils::ImageName& id) const {
    VkDescriptorImageInfo res{};
    const Texture* empty = getEmpty(id);
    if (empty) {
        res.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        res.imageView = empty->imageView();
        res.sampler = empty->sampler();
    } else {
        res.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        res.imageView = VK_NULL_HANDLE;
        res.sampler = VK_NULL_HANDLE;
    }
    return res;
}

}
