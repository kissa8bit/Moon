#ifndef MOON_ENTITIES_DIRECTIONAL_LIGHT_H
#define MOON_ENTITIES_DIRECTIONAL_LIGHT_H

#include <transformationals/lights.h>
#include <implementations/directionalLight.h>
#include <math/linearAlgebra.h>

namespace moon::entities {

class DirectionalLight : public transformational::Light {
private:
    static constexpr float kNear = 0.1f;
    float m_width{ 100.0f };
    float m_height{ 100.0f };
    float m_far{ 200.0f };
    implementations::DirectionalLight m_light;

public:
    interfaces::Light* light() override { return dynamic_cast<interfaces::Light*>(&m_light); }

    struct Props {
        bool enableShadow{ true };
        float power{ 10.0f };
        float drop{ 0.0f };
        float width{ 100.0f };
        float height{ 100.0f };
        float farPlane{ 200.0f };
    };

    DirectionalLight(const math::vec4& color = math::vec4(1.0f), const Props& props = {});
    ~DirectionalLight() override = default;

    DirectionalLight(const DirectionalLight&) = delete;
    DirectionalLight& operator=(const DirectionalLight&) = delete;

    DirectionalLight(DirectionalLight&&) = delete;
    DirectionalLight& operator=(DirectionalLight&&) = delete;

    DirectionalLight& setEnable(bool enable);
    bool isEnable() const;

    DirectionalLight& setColor(const math::vec4& color);
    DirectionalLight& setPower(float power);
    DirectionalLight& setDrop(float drop);
    DirectionalLight& setEnableShadow(bool enable);
    DirectionalLight& setWidth(float width);
    DirectionalLight& setHeight(float height);
    DirectionalLight& setFar(float far);
    DirectionalLight& setProjectionMatrix(const math::mat4& projection);

    math::vec4 getColor() const;
    float getPower() const;
    float getDrop() const;
    bool getEnableShadow() const;
    float getWidth() const;
    float getHeight() const;
    float getFar() const;
};

} // moon::entities

#endif // MOON_ENTITIES_DIRECTIONAL_LIGHT_H
