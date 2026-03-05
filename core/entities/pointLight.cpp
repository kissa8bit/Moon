#include "pointLight.h"

namespace moon::entities {

PointLight::PointLight(const Props& props) : transformational::Light(), m_light()
{
    setColor(props.color).setRadius(props.radius).setPower(props.power).setDrop(props.drop);
}

#define PointLightSetter(field, v)      \
m_light.buffer(true).field = v;         \
return *this;

PointLight& PointLight::setColor(const math::vec4& color) {
    PointLightSetter(color, color)
}

PointLight& PointLight::setRadius(float radius) {
    PointLightSetter(props.radius, radius)
}

PointLight& PointLight::setPower(float power) {
    PointLightSetter(props.powerFactor, power)
}

PointLight& PointLight::setDrop(float drop) {
    PointLightSetter(props.dropFactor, drop)
}

math::vec4 PointLight::getColor()  const { return m_light.buffer().color; }
float PointLight::getRadius()      const { return m_light.buffer().props.radius; }
float PointLight::getPower()       const { return m_light.buffer().props.powerFactor; }
float PointLight::getDrop()        const { return m_light.buffer().props.dropFactor; }

} // moon::entities
