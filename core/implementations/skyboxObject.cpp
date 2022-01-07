#include "skyboxObject.h"

namespace moon::implementations {

utils::vkDefault::DescriptorSetLayout SkyboxObject::createDescriptorSetLayout(VkDevice device) {
    std::vector<VkDescriptorSetLayoutBinding> binding;
    binding.push_back(utils::vkDefault::bufferVertexLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    binding.push_back(utils::vkDefault::imageFragmentLayoutBinding(static_cast<uint32_t>(binding.size()), 1));
    return utils::vkDefault::DescriptorSetLayout(device, binding);
}

SkyboxObject::SkyboxObject(const utils::vkDefault::Paths& texturePaths, const float& mipLevel)
    : interfaces::Object(interfaces::ObjectMask(interfaces::ObjectType::skybox, interfaces::ObjectProperty::enable)), uniformBuffer(&hostBuffer, sizeof(hostBuffer)), texturePaths(texturePaths)
{
    setMipLevel(mipLevel);
}

SkyboxObject& SkyboxObject::setMipLevel(float mipLevel) {
    texture.setMipLevel(mipLevel);
    return *this;
}

void SkyboxObject::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = createDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        auto descriptorSet = descriptors[i];
        const auto bufferInfo = uniformBuffer.device[i].descriptorBufferInfo();
        const auto imageInfo = texture.descriptorImageInfo();

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, descriptorSet, bufferInfo);
        utils::descriptorSet::write(writes, descriptorSet, imageInfo);
        utils::descriptorSet::update(device.device(), writes);
    }
}

void SkyboxObject::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

    VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
    texture = texturePaths.empty() ? utils::Texture::empty(device, commandBuffer) : utils::CubeTexture(texturePaths, device, device.device(), commandBuffer);
    CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
    texture.destroyCache();

    createDescriptors(device, imageCount);
}

void SkyboxObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

utils::Buffers& SkyboxObject::buffers() {
    return uniformBuffer.device;
}

SkyboxObject::Buffer& SkyboxObject::buffer(bool update) {
    if (update) {
        utils::vkDefault::raiseFlags(uniformBuffer.device);
    }
    return hostBuffer;
}

void SkyboxObject::setTransformation(const math::mat4& transformation) {
    buffer(true).modelMatrix = transpose(transformation);
}

} // moon::implementations