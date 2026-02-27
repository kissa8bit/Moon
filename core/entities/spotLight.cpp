#include "spotLight.h"

namespace moon::entities {

SpotLight::SpotLight(const Coloring& coloring, const Props& props, implementations::SpotLight::Type type)
	: transformational::Light(), m_fov(props.fov), m_aspect(props.aspect), m_far(props.farPlane)
{
	pLight = std::make_unique<implementations::SpotLight>(type);

	setDrop(props.drop).setPower(props.power).setInnerFraction(props.innerFraction).setExponent(props.exponent)
		.setColor(coloring.uniformColor).setProjectionMatrix(math::perspective(m_fov, m_aspect, kNear, m_far));

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

#define SpotLightGetter(field, defaultVal)                                                  \
if (auto pSpotLight = static_cast<implementations::SpotLight*>(pLight.get()); pSpotLight) {\
    return pSpotLight->buffer(false).props.field;                                           \
}                                                                                           \
return defaultVal;

float SpotLight::getDrop()         { SpotLightGetter(dropFactor,    0.0f) }
float SpotLight::getPower()        { SpotLightGetter(powerFactor,   0.0f) }
float SpotLight::getInnerFraction(){ SpotLightGetter(innerFraction, 0.0f) }
float SpotLight::getExponent()     { SpotLightGetter(exponent,      4.0f) }

SpotLight& SpotLight::setEnableShadow(bool enable) {
	if (pLight) pLight->lightMask().set(interfaces::LightProperty::enableShadow, enable);
	return *this;
}

SpotLight& SpotLight::setEnableScattering(bool enable) {
	if (pLight) pLight->lightMask().set(interfaces::LightProperty::enableScattering, enable);
	return *this;
}

bool SpotLight::getEnableShadow()     { return pLight ? pLight->lightMask().property().has(interfaces::LightProperty::enableShadow)     : false; }
bool SpotLight::getEnableScattering() { return pLight ? pLight->lightMask().property().has(interfaces::LightProperty::enableScattering) : false; }

SpotLight& SpotLight::setFov(float fov) {
	m_fov = fov;
	return setProjectionMatrix(math::perspective(m_fov, m_aspect, kNear, m_far));
}

SpotLight& SpotLight::setAspect(float aspect) {
	m_aspect = aspect;
	return setProjectionMatrix(math::perspective(m_fov, m_aspect, kNear, m_far));
}

SpotLight& SpotLight::setFar(float farPlane) {
	m_far = farPlane;
	return setProjectionMatrix(math::perspective(m_fov, m_aspect, kNear, m_far));
}

float SpotLight::getFov()    const { return m_fov; }
float SpotLight::getAspect() const { return m_aspect; }
float SpotLight::getFar()    const { return m_far; }

IsotropicLight::IsotropicLight(const Props& props) {
	const SpotLight::Coloring coloring(props.color);
	const SpotLight::Props spotProps{
		props.enableShadow, props.enableScattering,
		props.drop, props.power, props.innerFraction, props.exponent,
		math::radians(90.0f), 1.0f, props.radius
	};
	const auto type = implementations::SpotLight::Type::square;

	lights.reserve(6);
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(90.0f),  math::vec3(1.0f, 0.0f, 0.0f));
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(-90.0f), math::vec3(1.0f, 0.0f, 0.0f));
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(0.0f),   math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(90.0f),  math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(-90.0f), math::vec3(0.0f, 1.0f, 0.0f));
	lights.emplace_back(coloring, spotProps, type).rotate(math::radians(180.0f), math::vec3(1.0f, 0.0f, 0.0f));
	for (auto& light : lights) add(&light);

	// colors for debug if color = {0, 0, 0, 0}
	if (dot(props.color, props.color) == 0.0f && lights.size() == 6) {
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

IsotropicLight& IsotropicLight::setInnerFraction(const float& val) {
	GENERATE_SETTER(setInnerFraction)
}

IsotropicLight& IsotropicLight::setExponent(const float& val) {
	GENERATE_SETTER(setExponent)
}

float IsotropicLight::getDrop()          { return lights.empty() ? 0.0f  : lights.front().getDrop(); }
float IsotropicLight::getPower()         { return lights.empty() ? 0.0f  : lights.front().getPower(); }
float IsotropicLight::getInnerFraction() { return lights.empty() ? 0.0f  : lights.front().getInnerFraction(); }
float IsotropicLight::getExponent()      { return lights.empty() ? 4.0f  : lights.front().getExponent(); }

IsotropicLight& IsotropicLight::setEnableShadow(bool val) {
	GENERATE_SETTER(setEnableShadow)
}

IsotropicLight& IsotropicLight::setEnableScattering(bool val) {
	GENERATE_SETTER(setEnableScattering)
}

bool IsotropicLight::getEnableShadow()     { return lights.empty() ? false : lights.front().getEnableShadow(); }
bool IsotropicLight::getEnableScattering() { return lights.empty() ? false : lights.front().getEnableScattering(); }

std::vector<interfaces::Light*> IsotropicLight::getLights() const {
	std::vector<interfaces::Light*> pLights;
	for (const auto& light : lights) {
		pLights.push_back(static_cast<interfaces::Light*>(light));
	}
	return pLights;
}

} // moon::entities
