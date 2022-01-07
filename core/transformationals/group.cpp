#include "group.h"

namespace moon::transformational {

Group& Group::update() {
    math::mat4 transformMatrix = convert(convert(m_rotation, m_translation));
    math::mat4 modelMatrix = m_globalTransformation * transformMatrix * moon::math::scale(m_scaling);
    for (auto& object : objects) {
        object->setGlobalTransform(modelMatrix);
    }
    return *this;
}

DEFAULT_TRANSFORMATIONAL_DEFINITION(Group)
DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Group)

bool Group::add(Transformational* object) {
    auto [_, inserted] = objects.insert(object);
    if (inserted) update();
    return inserted;
}

bool Group::remove(Transformational* object) {
    return objects.erase(object);
}

bool Group::find(Transformational* object) {
    return objects.find(object) != objects.end();
}

} // moon::transformational
