#include "spotLight.h"

namespace moon::entities {

SpotLight::SpotLight(const Coloring& coloring, const math::mat4& projection, const Props& props, implementations::SpotLight::Type type)
	: transformational::Light()
{
	pLight = std::make_unique<implementations::SpotLight>(type);

	setDrop(props.drop).setPower(props.power).setInnerFraction(props.innerFraction).setExponent(props.exponent).setColor(coloring.uniformColor).setProjectionMatrix(projection);

	auto pSpotLight = static_cast<implementations::SpotLight*>(pLight.get());
	pSpotLight->setTexture(coloring.texturePath);
	interfaces::LightProperty properties;
	properties.set(interfaces::LightProperty::enableScattering, props.enableScattering);
	properties.set(interfaces::LightProperty::enableShadow, props.enableShadow);
	pSpotLight->lightMask().set(properties, true);
}

#define SpotLightSetter(field, v)																	\
if (auto pSpotLight = static_cast<implementations::SpotLight*>(pLight.get()); pSpotLight) {			\
    pSpotLight->buffer(true).field = v;																\
}																									\
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

SpotLight& SpotLight::setInnerFraction(const float& innerFraction) {
	SpotLightSetter(props.innerFraction, innerFraction)
}

SpotLight& SpotLight::setExponent(const float& exponent) {
	SpotLightSetter(props.exponent, exponent)
}

IsotropicLight::IsotropicLight(const math::vec4& color, float radius, bool enableShadow, bool enableScattering) {
	const auto proj = math::perspective(math::radians(91.0f), 1.0f, 0.1f, radius);

	const SpotLight::Coloring coloring(color);
	const SpotLight::Props props{ enableShadow, enableScattering };
	const auto type = implementations::SpotLight::Type::square;

	lights.reserve(6);
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(90.0f), math::vec3(1.0f, 0.0f, 0.0f));
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(-90.0f), math::vec3(1.0f, 0.0f, 0.0f));
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(0.0f), math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(90.0f), math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(-90.0f), math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, proj, props, type).rotate(math::radians(180.0f), math::vec3(1.0f, 0.0f, 0.0f));
	for (auto& light : lights) add(&light);

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

} // moon::entities
