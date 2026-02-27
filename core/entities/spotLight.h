#ifndef MOON_ENTITIES_SPOT_LIGHT_H
#define MOON_ENTITIES_SPOT_LIGHT_H

#include <vector>

#include <transformationals/lights.h>
#include <transformationals/group.h>

#include <implementations/spotLight.h>

#include <math/linearAlgebra.h>

namespace moon::entities {

class SpotLight : public transformational::Light {
public:
    struct Coloring
    {
        math::vec4 uniformColor{ 0.0f };
        std::filesystem::path texturePath{};

        Coloring(const math::vec4& uniformColor) : uniformColor(uniformColor) {}
        Coloring(const std::filesystem::path& texturePath) : texturePath(texturePath) {}
    };

    struct Props
    {
        bool enableShadow{ true };
        bool enableScattering{ false };
        float drop{ 1.0f };
        float power{ 10.0f };
        float innerFraction{ 1.0f };
        float exponent{ 4.0f };
    };

    SpotLight(const Coloring& coloring, const math::mat4& projection, const Props& props = {}, implementations::SpotLight::Type type = implementations::SpotLight::Type::circle);
    ~SpotLight() override = default;

    SpotLight(const SpotLight&) = delete;
    SpotLight& operator=(const SpotLight&) = delete;

    SpotLight(SpotLight&&) = default;
    SpotLight& operator=(SpotLight&&) = default;

    SpotLight& setColor(const math::vec4& color);
    SpotLight& setDrop(const float& drop);
    SpotLight& setPower(const float& power);
    SpotLight& setInnerFraction(const float& innerFraction);
    SpotLight& setExponent(const float& exponent);
    SpotLight& setProjectionMatrix(const math::mat4& projection);
};

class IsotropicLight : public transformational::Group {
private:
    std::vector<SpotLight> lights;

public:
    IsotropicLight(const math::vec4& color = { 0.0f }, float radius = 100.0f, bool enableShadow = true, bool enableScattering = false);
    ~IsotropicLight() = default;

    IsotropicLight& setColor(const math::vec4& color);
    IsotropicLight& setDrop(const float& drop);
    IsotropicLight& setPower(const float& power);
    IsotropicLight& setProjectionMatrix(const math::mat4& projection);

    std::vector<interfaces::Light*> getLights() const;
};

} // moon::entities

#endif // MOON_ENTITIES_SPOT_LIGHT_H
