#include "cameras.h"

#include <cstring>

namespace moon::transformational {

Camera& Camera::update() {
	const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation));
	camera()->setTransformation(transformMatrix);
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Camera)

}
