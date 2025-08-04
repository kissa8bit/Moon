#include "objects.h"

#include <cstring>

#include <utils/operations.h>
#include <utils/device.h>

#include <implementations/baseObject.h>
#include <implementations/skyboxObject.h>

namespace moon::transformational {

Object& Object::update() {
	if (pObject) {
		const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation)) * math::scale(m_scaling);
		pObject->setTransformation(transformMatrix);
	}
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Object)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Object)

Object::operator interfaces::Object* () const {
    return pObject.get();
}

}
