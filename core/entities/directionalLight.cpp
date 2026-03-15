#include "directionalLight.h"

namespace moon::entities {

// math::orthographic() produces OpenGL-style depth NDC in [-1, 1].
// Vulkan clips fragments with z_ndc < 0, so objects in the near half of the frustum
// are never written to the shadow map. Remap to Vulkan [0, 1]: z_vk = 0.5*z_gl + 0.5.
static math::mat4 makeOrthoProj(float width, float height, float n, float f) {
    auto m = math::orthographic(width, height, n, f);
    // Remap z from OpenGL [-1,1] to Vulkan [0,1]: z_vk = 0.5*z_gl + 0.5
    m[2][2] *= 0.5f;
    m[2][3] = m[2][3] * 0.5f + 0.5f;
    return m;
}

DirectionalLight::DirectionalLight(const math::vec4& color, const Props& props)
    : transformational::Light(), m_light(), m_width(props.width), m_height(props.height), m_far(props.farPlane)
{
    setPower(props.power).setDrop(props.drop).setColor(color).setProjectionMatrix(makeOrthoProj(m_width, m_height, kNear, m_far));

    interfaces::LightProperty properties;
    properties.set(interfaces::LightProperty::enableShadow, props.enableShadow);
    m_light.lightMask().set(properties, true);
}

DirectionalLight& DirectionalLight::setEnable(bool enable) { m_light.setEnable(enable); return *this; }
bool              DirectionalLight::isEnable() const       { return m_light.isEnable(); }

#define DirLightSetter(field, v)    \
m_light.buffer(true).field = v;    \
return *this;

DirectionalLight& DirectionalLight::setProjectionMatrix(const math::mat4& projection) {
    DirLightSetter(proj, transpose(projection))
}

DirectionalLight& DirectionalLight::setColor(const math::vec4& color) {
    DirLightSetter(color, color)
}

DirectionalLight& DirectionalLight::setPower(float power) {
    DirLightSetter(props.powerFactor, power)
}

DirectionalLight& DirectionalLight::setDrop(float drop) {
    DirLightSetter(props.dropFactor, drop)
}

DirectionalLight& DirectionalLight::setEnableShadow(bool enable) {
    m_light.lightMask().set(interfaces::LightProperty::enableShadow, enable);
    return *this;
}

DirectionalLight& DirectionalLight::setWidth(float width) {
    m_width = width;
    return setProjectionMatrix(makeOrthoProj(m_width, m_height, kNear, m_far));
}

DirectionalLight& DirectionalLight::setHeight(float height) {
    m_height = height;
    return setProjectionMatrix(makeOrthoProj(m_width, m_height, kNear, m_far));
}

DirectionalLight& DirectionalLight::setFar(float far) {
    m_far = far;
    return setProjectionMatrix(makeOrthoProj(m_width, m_height, kNear, m_far));
}

math::vec4 DirectionalLight::getColor() const    { return m_light.buffer().color; }
float      DirectionalLight::getPower() const    { return m_light.buffer().props.powerFactor; }
float      DirectionalLight::getDrop() const     { return m_light.buffer().props.dropFactor; }
float      DirectionalLight::getWidth() const    { return m_width; }
float      DirectionalLight::getHeight() const   { return m_height; }
float      DirectionalLight::getFar() const      { return m_far; }

bool DirectionalLight::getEnableShadow() const {
    return m_light.lightMask().property().has(interfaces::LightProperty::enableShadow);
}

} // moon::entities
