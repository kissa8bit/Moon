#ifndef MOON_ENTITIES_SKYBOX_OBJECT_H
#define MOON_ENTITIES_SKYBOX_OBJECT_H

#include <optional>

#include <transformationals/objects.h>

namespace moon::entities {

class SkyboxObject : public transformational::Object {
public:
    SkyboxObject(const utils::vkDefault::Paths& texturePaths, const float& mipLevel = 1.0f);

    SkyboxObject(const SkyboxObject&) = delete;
    SkyboxObject& operator=(const SkyboxObject&) = delete;

    SkyboxObject(SkyboxObject&&) = default;
    SkyboxObject& operator=(SkyboxObject&&) = default;

    SkyboxObject& setColor(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);
};

} // moon::entities

#endif // MOON_ENTITIES_SKYBOX_OBJECT_H