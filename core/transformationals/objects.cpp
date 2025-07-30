#include "objects.h"

#include "operations.h"
#include "model.h"
#include "device.h"

#include <cstring>

namespace moon::interfaces {

BaseObject::BaseObject(ObjectMask objectMask, void* hostData, size_t hostDataSize)
    : Object(objectMask), uniformBuffer(hostData, hostDataSize)
{}

BaseObject::BaseObject(ObjectMask objectMask, void* hostData, size_t hostDataSize, interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount)
    : Object(objectMask, model, {firstInstance, instanceCount}), uniformBuffer(hostData, hostDataSize)
{}

void BaseObject::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void BaseObject::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Object::createBaseDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptors = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);
    for (size_t i = 0; i < imageCount; i++) {
        const auto bufferInfo = uniformBuffer.device[i].descriptorBufferInfo();

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, descriptors[i], bufferInfo);
        utils::descriptorSet::update(device.device(), writes);
    }
}

void BaseObject::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);
    createDescriptors(device, imageCount);
}

utils::Buffers& BaseObject::buffers() {
    return uniformBuffer.device;
}

SkyboxObject::SkyboxObject(ObjectMask objectMask, void* hostData, size_t hostDataSize, const utils::vkDefault::Paths& texturePaths, const float& mipLevel)
    : BaseObject(objectMask, hostData, hostDataSize), texturePaths(texturePaths) {
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

}

namespace moon::transformational {

Object::Object(interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount) {
    pObject = std::make_unique<interfaces::BaseObject>(interfaces::ObjectMask(interfaces::ObjectType::base), &buffer, sizeof(buffer), model, firstInstance, instanceCount);
    pObject->objectMask().set(interfaces::ObjectProperty::enable, true);
    pObject->objectMask().set(interfaces::ObjectProperty::enableShadow, true);

    if (model) {
        for (auto instance = 0; instance < instanceCount; ++instance) {
            animationControl.animationsMap[instance] = model->animations(firstInstance + instance);
            animationControl.total = animationControl.animationsMap[instance].size();
        }
    }
}

Object::Object(const utils::vkDefault::Paths& texturePaths, const float& mipLevel) {
    pObject = std::make_unique<interfaces::SkyboxObject>(interfaces::ObjectMask(interfaces::ObjectType::skybox), &buffer, sizeof(buffer), texturePaths, mipLevel);
    pObject->objectMask().set(interfaces::ObjectProperty::enable, true);
}

Object& Object::update() {
    math::mat4 transformMatrix = convert(convert(m_rotation, m_translation));
    buffer.modelMatrix = transpose(m_globalTransformation * transformMatrix * math::scale(m_scaling));
    utils::vkDefault::raiseFlags(pObject->buffers());
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Object)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Object)

Object& Object::setBase(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    if (constant.has_value()) {
        buffer.base.constant = constant.value();
    }
    if (factor.has_value()) {
        buffer.base.factor = factor.value();
    }
    utils::vkDefault::raiseFlags(pObject->buffers());
    return *this;
}

Object& Object::setBloom(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    buffer.bloom.constant = constant.value_or(buffer.bloom.constant);
    buffer.bloom.factor = factor.value_or(buffer.bloom.factor);
    utils::vkDefault::raiseFlags(pObject->buffers());
    return *this;
}

Object& Object::setOutlining(const bool enable, const float width, const math::vec4& color) {
    buffer.outlining.width = width > 0.0f ? width : buffer.outlining.width;
    buffer.outlining.color = math::dot(color, color) > 0.0f ? color : buffer.outlining.color;
    utils::vkDefault::raiseFlags(pObject->buffers());
    pObject->objectMask().set(interfaces::ObjectType::outlining, enable);
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
        if (index < static_cast<int>(animations.size())) {
            animIndex = index;
            startOffset = changeTime;
            time = 0;
            if (animIndex > -1) {
                animations.at(animIndex)->setChangeTime(changeTime);
            }
        }
    }
}

bool Object::AnimationControl::update(size_t frameNumber, float dtime) {
    if (auto animationsIt = animationsMap.find(frameNumber); animationsIt != animationsMap.end()) {
        auto& [_, animations] = *animationsIt;

        if (animations.size() == 0 || animIndex < 0) return false;

        auto animation = animations.at(animIndex);
        if (!animation) return false;

        time += dtime;
        if (time > animation->duration()) {
            time = startOffset;
        }
        return animation->update(time);
    }
    return false;
}

}
