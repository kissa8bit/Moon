#include "lights.h"

#include <cstring>

namespace moon::transformational {

Light& Light::update() {
	const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation)) * math::scale(m_scaling);
	light()->setTransformation(transformMatrix);
	return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Light)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Light)

} // moon::transformational
