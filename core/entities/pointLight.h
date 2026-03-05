#ifndef MOON_ENTITIES_POINT_LIGHT_H
#define MOON_ENTITIES_POINT_LIGHT_H

#include <transformationals/lights.h>

#include <implementations/pointLight.h>

#include <math/linearAlgebra.h>

namespace moon::entities {

class PointLight : public transformational::Light {
private:
    implementations::PointLight m_light;

public:
    interfaces::Light* light() override { return dynamic_cast<interfaces::Light*>(&m_light); }

    struct Props {
        math::vec4 color{ 0.0f };
        float radius{ 10.0f };
        float power{ 10.0f };
        float drop{ 1.0f };
    };

    PointLight(const Props& props = {});
    ~PointLight() override = default;

    PointLight(const PointLight&) = delete;
    PointLight& operator=(const PointLight&) = delete;

    PointLight(PointLight&&) = delete;
    PointLight& operator=(PointLight&&) = delete;

    PointLight& setColor(const math::vec4& color);
    PointLight& setRadius(float radius);
    PointLight& setPower(float power);
    PointLight& setDrop(float drop);

    math::vec4 getColor() const;
    float getRadius() const;
    float getPower() const;
    float getDrop() const;
};

} // moon::entities

#endif // MOON_ENTITIES_POINT_LIGHT_H
