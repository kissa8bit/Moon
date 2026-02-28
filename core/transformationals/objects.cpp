#include "objects.h"

#include <cstring>

namespace moon::transformational {

Object& Object::update() {
	const math::mat4 transformMatrix = m_globalTransformation * convert(convert(m_rotation, m_translation)) * math::scale(m_scaling);
	object()->setTransformation(transformMatrix);
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Object)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Object)

}
