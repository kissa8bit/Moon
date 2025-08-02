#include "lights.h"

#include "operations.h"
#include "device.h"

#include <cstring>

namespace {
	moon::interfaces::LightType toLightType(moon::interfaces::SpotLight::Type type)
	{
		moon::interfaces::LightType res{};
		switch (type)
		{
		case moon::interfaces::SpotLight::Type::circle:
			res = moon::interfaces::LightType::spotCircle;
			break;
		case moon::interfaces::SpotLight::Type::square:
			res = moon::interfaces::LightType::spotSquare;
			break;
		}
		return res;
	}
}

namespace moon::interfaces {

SpotLight::SpotLight(Type type)
	: Light(LightMask(toLightType(type))), uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void SpotLight::setTexture(const std::filesystem::path& path) {
	texturePath = path;
}

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
		utils::descriptorSet::Writes writes;
		WRITE_DESCRIPTOR(writes, descriptorSets.at(i), uniformBuffer.device[i].descriptorBufferInfo());
		WRITE_DESCRIPTOR(writes, descriptorSets.at(i), texture.descriptorImageInfo());
		utils::descriptorSet::update(device.device(), writes);
	}
}

utils::Buffers& SpotLight::buffers() {
	return uniformBuffer.device;
}

SpotLight::Buffer& SpotLight::buffer(bool update) {
	if (update) {
		utils::vkDefault::raiseFlags(uniformBuffer.device);
	}
	return hostBuffer;
}

void SpotLight::setTransformation(const math::mat4& transformation) {
	buffer(true).view = transpose(inverse(transformation));
}

}

namespace moon::transformational {

Light::Light(interfaces::LightType type) {
	switch (type)
	{
	case interfaces::LightType::spotCircle:
		pLight = std::make_unique<interfaces::SpotLight>(interfaces::SpotLight::Type::circle);
		break;
	case interfaces::LightType::spotSquare:
		pLight = std::make_unique<interfaces::SpotLight>(interfaces::SpotLight::Type::square);
		break;
	}
}

Light& Light::update() {
	const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation)) * math::scale(m_scaling);
	pLight->setTransformation(transformMatrix);
	return *this;
}

Light::operator interfaces::Light* () const {
	return pLight.get();
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Light)

}

namespace moon::entities {

SpotLight::SpotLight(const Coloring& coloring, const math::mat4& projection, const Props& props, interfaces::SpotLight::Type type)
	: transformational::Light(toLightType(type))
{
	setDrop(props.drop).setPower(props.power).setColor(coloring.uniformColor).setProjectionMatrix(projection);

	auto pSpotLight = static_cast<interfaces::SpotLight*>(pLight.get());
	if (!pSpotLight) return;

	pSpotLight->setTexture(coloring.texturePath);
	interfaces::LightProperty properties;
	properties.set(interfaces::LightProperty::enableScattering, props.enableScattering);
	properties.set(interfaces::LightProperty::enableShadow, props.enableShadow);
	pSpotLight->lightMask().set(properties, true);
}

#define SpotLightSetter(field, v)                                                               \
    if (auto pSpotLight = static_cast<interfaces::SpotLight*>(pLight.get()); pSpotLight) {      \
        pSpotLight->buffer(true).field = v;                                                     \
    }                                                                                           \
    return *this;

SpotLight& SpotLight::setProjectionMatrix(const math::mat4& projection) {
	SpotLightSetter(proj, transpose(projection))
}

SpotLight& SpotLight::setColor(const math::vec4& color) {
	SpotLightSetter(color, color)
}

SpotLight& SpotLight::setDrop(const float& drop) {
	SpotLightSetter(props.dropFactor, drop)
}

SpotLight& SpotLight::setPower(const float& power) {
	SpotLightSetter(props.powerFactor, power)
}

IsotropicLight::IsotropicLight(const math::vec4& color, float radius, bool enableShadow, bool enableScattering) {
	const auto proj = math::perspective(math::radians(91.0f), 1.0f, 0.1f, radius);

	lights.reserve(6);

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(90.0f), math::vec3(1.0f, 0.0f, 0.0f)));

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(-90.0f), math::vec3(1.0f, 0.0f, 0.0f)));

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(0.0f), math::vec3(0.0f, 1.0f, 0.0f)));

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(90.0f), math::vec3(0.0f, 1.0f, 0.0f)));

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(-90.0f), math::vec3(0.0f, 1.0f, 0.0f)));

	add(&lights.emplace_back(SpotLight(color, proj, {enableShadow, enableScattering}, interfaces::SpotLight::Type::square))
		.rotate(math::radians(180.0f), math::vec3(1.0f, 0.0f, 0.0f)));

	// colors for debug if color = {0, 0, 0, 0}
	if (dot(color, color) == 0.0f && lights.size() == 6) {
		lights.at(0).setColor(math::vec4(1.0f, 0.0f, 0.0f, 1.0f));
		lights.at(1).setColor(math::vec4(0.0f, 1.0f, 0.0f, 1.0f));
		lights.at(2).setColor(math::vec4(0.0f, 0.0f, 1.0f, 1.0f));
		lights.at(3).setColor(math::vec4(0.3f, 0.6f, 0.9f, 1.0f));
		lights.at(4).setColor(math::vec4(0.6f, 0.9f, 0.3f, 1.0f));
		lights.at(5).setColor(math::vec4(0.9f, 0.3f, 0.6f, 1.0f));
	}
}

#define GENERATE_SETTER(func)					\
for (auto& light : lights) light.func(val);		\
return *this;

IsotropicLight& IsotropicLight::setProjectionMatrix(const math::mat4& val) {
	GENERATE_SETTER(setProjectionMatrix)
}

IsotropicLight& IsotropicLight::setColor(const math::vec4& val) {
	GENERATE_SETTER(setColor)
}

IsotropicLight& IsotropicLight::setDrop(const float& val) {
	GENERATE_SETTER(setDrop)
}

IsotropicLight& IsotropicLight::setPower(const float& val) {
	GENERATE_SETTER(setPower)
}

std::vector<interfaces::Light*> IsotropicLight::getLights() const {
	std::vector<interfaces::Light*> pLights;
	for (const auto& light : lights) {
		pLights.push_back(static_cast<interfaces::Light*>(light));
	}
	return pLights;
}

}
