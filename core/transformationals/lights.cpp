#include "lights.h"

#include <implementations/spotLight.h>

#include <cstring>

namespace moon::transformational {

Light::Light(interfaces::LightType type) {
	switch (type)
	{
	case interfaces::LightType::spotCircle:
		pLight = std::make_unique<implementations::SpotLight>(implementations::SpotLight::Type::circle);
		break;
	case interfaces::LightType::spotSquare:
		pLight = std::make_unique<implementations::SpotLight>(implementations::SpotLight::Type::square);
		break;
	}
}

Light& Light::update() {
	if (pLight) {
		const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation)) * math::scale(m_scaling);
		pLight->setTransformation(transformMatrix);
	}
	return *this;
}

Light::operator interfaces::Light* () const {
	return pLight.get();
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Light)

} // moon::transformational
