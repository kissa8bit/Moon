#include "objects.h"

#include "operations.h"
#include "model.h"
#include "dualQuaternion.h"
#include "device.h"

#include <cstring>

namespace moon::interfaces {

BaseObject::BaseObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize)
    : Object(pipelineBitMask), uniformBuffer(hostData, hostDataSize)
{}

BaseObject::BaseObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize, interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount)
    : Object(pipelineBitMask, model, firstInstance, instanceCount), uniformBuffer(hostData, hostDataSize)
{}

void BaseObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseObject::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Object::createBaseDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
    for (size_t i = 0; i < imageCount; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer.device[i];
        bufferInfo.offset = 0;
        bufferInfo.range = uniformBuffer.size;

        std::vector<VkWriteDescriptorSet> descriptorWrites{};
        descriptorWrites.push_back(VkWriteDescriptorSet{});
        descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
        descriptorWrites.back().dstSet = descriptors[i];
        descriptorWrites.back().descriptorCount = 1;
        descriptorWrites.back().pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device.device(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void BaseObject::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
    createDescriptors(device, imageCount);
}

utils::Buffers& BaseObject::buffers() {
    return uniformBuffer.device;
}

SkyboxObject::SkyboxObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize, const utils::Paths& texturePaths, const float& mipLevel)
    : BaseObject(pipelineBitMask, hostData, hostDataSize), texturePaths(texturePaths) {
    setMipLevel(mipLevel);
}

SkyboxObject& SkyboxObject::setMipLevel(float mipLevel) {
    texture.setMipLevel(mipLevel);
    return *this;
}

void SkyboxObject::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Object::createSkyboxDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer.device[i];
        bufferInfo.offset = 0;
        bufferInfo.range = uniformBuffer.size;

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = texture.imageView();
        imageInfo.sampler = texture.sampler();

        std::vector<VkWriteDescriptorSet> descriptorWrites;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pBufferInfo = &bufferInfo;
        descriptorWrites.push_back(VkWriteDescriptorSet{});
            descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.back().dstSet = descriptors[i];
            descriptorWrites.back().dstBinding = static_cast<uint32_t>(descriptorWrites.size() - 1);
            descriptorWrites.back().dstArrayElement = 0;
            descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites.back().descriptorCount = 1;
            descriptorWrites.back().pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(device.device(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void SkyboxObject::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount)
{
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

    VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
    texture = texturePaths.empty() ? utils::Texture::empty(device, commandBuffer) : utils::CubeTexture(texturePaths, device, device.device(), commandBuffer);
    CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
    texture.destroyCache();

    createDescriptors(device, imageCount);
}

}

namespace moon::transformational {

Object::Object(interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount) {
    uint8_t pipelineBitMask = interfaces::ObjectType::base | interfaces::ObjectProperty::non;
    pObject = std::make_unique<interfaces::BaseObject>(pipelineBitMask , &buffer, sizeof(buffer), model, firstInstance, instanceCount);

    if (model) {
        for (auto instance = 0; instance < instanceCount; ++instance) {
            animationControl.animationsMap[instance] = model->animations(firstInstance + instance);
            animationControl.total = animationControl.animationsMap[instance].size();
        }
    }
}

Object::Object(const utils::Paths& texturePaths, const float& mipLevel) {
    uint8_t pipelineBitMask = interfaces::ObjectType::skybox | interfaces::ObjectProperty::non;
    pObject = std::make_unique<interfaces::SkyboxObject>(pipelineBitMask, &buffer, sizeof(buffer), texturePaths, mipLevel);
}

Object& Object::update() {
    math::Matrix<float,4,4> transformMatrix = convert(convert(m_rotation, m_translation));
    buffer.modelMatrix = transpose(m_globalTransformation * transformMatrix * math::scale(m_scaling));
    utils::raiseFlags(pObject->buffers());
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Object)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Object)

Object& Object::setBase(std::optional<math::Vector<float, 4>> constant, std::optional<math::Vector<float, 4>> factor) {
    if (constant.has_value()) {
        buffer.base.constant = constant.value();
    }
    if (factor.has_value()) {
        buffer.base.factor = factor.value();
    }
    utils::raiseFlags(pObject->buffers());
    return *this;
}

Object& Object::setBloom(std::optional<math::Vector<float, 4>> constant, std::optional<math::Vector<float, 4>> factor) {
    buffer.bloom.constant = constant.value_or(buffer.bloom.constant);
    buffer.bloom.factor = factor.value_or(buffer.bloom.factor);
    utils::raiseFlags(pObject->buffers());
    return *this;
}

Object& Object::setOutlining(const bool& enable, const float& width, const math::Vector<float, 4>& color) {
    buffer.outlining.width = width > 0.0f ? width : buffer.outlining.width;
    buffer.outlining.color = math::dot(color, color) > 0.0f ? color : buffer.outlining.color;
    utils::raiseFlags(pObject->buffers());

    auto& pipelineFlagBits = pObject->pipelineFlagBits();
    pipelineFlagBits &= ~interfaces::ObjectProperty::outlining;
    if (enable) {
        pipelineFlagBits |= interfaces::ObjectProperty::outlining;
    }
    return *this;
}

Object::operator interfaces::Object* () const {
    return pObject.get();
}

size_t Object::AnimationControl::size() const {
    return total;
}

size_t Object::AnimationControl::current() const {
    return animIndex;
}

void Object::AnimationControl::set(int index, float changeTime) {
    for (auto& [_, animations] : animationsMap) {
        if (index > -1 && index < animations.size()) {
            animIndex = index;
            startOffset = changeTime;
            time = 0;
        }
    }
}

bool Object::AnimationControl::update(size_t frameNumber, float dtime) {
    if(animIndex == -1) return false;

    if (auto animationsIt = animationsMap.find(frameNumber); animationsIt != animationsMap.end()) {
        auto& [_, animations] = *animationsIt;
        auto animation = animations.at(animIndex);
        if (!animation) return false;

        time += dtime;
        if (!animation->change(time, startOffset)) {
            float animationTime = time - startOffset;
            if (animationTime > animation->duration()) {
                time = startOffset;
            }
            return animation->update(animationTime);
        }
    }
    return false;
}

}
