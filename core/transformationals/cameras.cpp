#include "cameras.h"

#include <cstring>

#include <utils/operations.h>
#include <utils/device.h>

#include <implementations/baseCamera.h>

namespace moon::transformational {

Camera& Camera::update() {
	if (pCamera) {
		const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation));
		pCamera->setTransformation(transformMatrix);
	}
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Camera)
DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Camera)

Camera::operator interfaces::Camera* () const {
    return pCamera.get();
}

}
