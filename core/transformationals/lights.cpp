#include "lights.h"

#include "operations.h"
#include "device.h"

#include <cstring>

namespace moon::interfaces {

SpotLight::SpotLight(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize, bool enableShadow, bool enableScattering, const std::filesystem::path& texturePath)
    : Light(pipelineBitMask, enableShadow, enableScattering), uniformBuffer(hostData, hostDataSize), texturePath(texturePath) {}

void SpotLight::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
    uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

    VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
    texture = texturePath.empty() ? utils::Texture::empty(device, commandBuffer) : utils::Texture(texturePath, device, device.device(), commandBuffer);
    CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
    texture.destroyCache();

    createDescriptors(device, imageCount);
}

void SpotLight::render(
    uint32_t frameNumber,
    VkCommandBuffer commandBuffer,
    const utils::vkDefault::DescriptorSets& descriptorSet,
    VkPipelineLayout pipelineLayout,
    VkPipeline pipeline)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    utils::vkDefault::DescriptorSets descriptors = descriptorSet;
    descriptors.push_back(descriptorSets[frameNumber]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptors.size()), descriptors.data(), 0, nullptr);
    vkCmdDraw(commandBuffer, 18, 1, 0, 0);
}

void SpotLight::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
    uniformBuffer.update(frameNumber, commandBuffer);
}

void SpotLight::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
    descriptorSetLayout = interfaces::Light::createDescriptorSetLayout(device.device());
    descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
    descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

    for (size_t i = 0; i < imageCount; i++) {
        auto descriptorSet = descriptorSets[i];
        const auto bufferInfo = uniformBuffer.device[i].descriptorBufferInfo();
        const auto imageInfo = texture.descriptorImageInfo();

        utils::descriptorSet::Writes writes;
        utils::descriptorSet::write(writes, descriptorSet, bufferInfo);
        utils::descriptorSet::write(writes, descriptorSet, imageInfo);
        utils::descriptorSet::update(device.device(), writes);
    }
}

utils::Buffers& SpotLight::buffers() {
    return uniformBuffer.device;
}

}

namespace moon::transformational {

Light::Light(const math::vec4& color, const math::mat4& projection, bool enableShadow, bool enableScattering, interfaces::SpotLight::Type type)
    : type(type)
{
    buffer.color = color;
    buffer.proj = transpose(projection);
    uint8_t pipelineBitMask = interfaces::Light::Type::spot;
    pLight = std::make_unique<interfaces::SpotLight>(pipelineBitMask, &buffer, sizeof(buffer), enableShadow, enableScattering);
}

Light::Light(const std::filesystem::path& texturePath, const math::mat4& projection, bool enableShadow, bool enableScattering, interfaces::SpotLight::Type type):
    type(type)
{
    buffer.proj = transpose(projection);
    uint8_t pipelineBitMask = interfaces::Light::Type::spot;
    pLight = std::make_unique<interfaces::SpotLight>(pipelineBitMask, &buffer, sizeof(buffer), enableShadow, enableScattering, texturePath);
}

Light& Light::update() {
    math::mat4 transformMatrix = convert(convert(m_rotation, m_translation));
    buffer.view = transpose(inverse(math::mat4(m_globalTransformation * transformMatrix * math::scale(m_scaling))));
    buffer.props = moon::math::vec4(static_cast<float>(type), lightPowerFactor, lightDropFactor, 0.0f);
    utils::vkDefault::raiseFlags(pLight->buffers());
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Light)

Light& Light::setProjectionMatrix(const math::mat4& projection)  {
    buffer.proj = transpose(projection);
    return update();
}

Light& Light::setColor(const math::vec4& color){
    buffer.color = color;
    return update();
}

Light& Light::setDrop(const float& drop) {
    lightDropFactor = drop;
    return update();
}

Light& Light::setPower(const float& power) {
    lightPowerFactor = power;
    return update();
}

Light::operator interfaces::Light* () const {
    return pLight.get();
}

IsotropicLight::IsotropicLight(const math::vec4& color, float radius, bool enableShadow, bool enableScattering) {
    const auto proj = math::perspective(math::radians(91.0f), 1.0f, 0.1f, radius);

    lights.reserve(6);

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(90.0f), math::vec3(1.0f, 0.0f, 0.0f)));

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(-90.0f), math::vec3(1.0f, 0.0f, 0.0f)));

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(0.0f), math::vec3(0.0f, 1.0f, 0.0f)));

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(90.0f), math::vec3(0.0f, 1.0f, 0.0f)));

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(-90.0f), math::vec3(0.0f, 1.0f, 0.0f)));

    add(&lights.emplace_back(std::make_unique<Light>(color, proj, enableShadow, enableScattering, interfaces::SpotLight::Type::square))
        ->rotate(math::radians(180.0f), math::vec3(1.0f, 0.0f, 0.0f)));

    // colors for debug if color = {0, 0, 0, 0}
    if(dot(color, color) == 0.0f && lights.size() == 6) {
        lights.at(0)->setColor(math::vec4(1.0f,0.0f,0.0f,1.0f));
        lights.at(1)->setColor(math::vec4(0.0f,1.0f,0.0f,1.0f));
        lights.at(2)->setColor(math::vec4(0.0f,0.0f,1.0f,1.0f));
        lights.at(3)->setColor(math::vec4(0.3f,0.6f,0.9f,1.0f));
        lights.at(4)->setColor(math::vec4(0.6f,0.9f,0.3f,1.0f));
        lights.at(5)->setColor(math::vec4(0.9f,0.3f,0.6f,1.0f));
    }
}

#define GENERATE_SETTER(func)       \
    for (auto& light : lights) {    \
        light->func(val);           \
    }                               \
    return *this;

IsotropicLight& IsotropicLight::setProjectionMatrix(const math::mat4& val){
    GENERATE_SETTER(setProjectionMatrix)
}

IsotropicLight& IsotropicLight::setColor(const math::vec4& val){
    GENERATE_SETTER(setColor)
}

IsotropicLight& IsotropicLight::setDrop(const float& val){
    GENERATE_SETTER(setDrop)
}

IsotropicLight& IsotropicLight::setPower(const float& val) {
    GENERATE_SETTER(setPower)
}

#undef GENERATE_SETTER

std::vector<interfaces::Light*> IsotropicLight::getLights() const {
    std::vector<interfaces::Light*> pLights;
    for (const auto& light: lights) {
        pLights.push_back(*light.get());
    }
    return pLights;
}

}
