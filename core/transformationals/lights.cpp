#include "lights.h"

#include <implementations/spotLight.h>

#include <cstring>

namespace moon::transformational {

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
