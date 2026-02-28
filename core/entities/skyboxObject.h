#ifndef MOON_ENTITIES_SKYBOX_OBJECT_H
#define MOON_ENTITIES_SKYBOX_OBJECT_H

#include <optional>

#include <transformationals/objects.h>
#include <implementations/skyboxObject.h>

namespace moon::entities {

class SkyboxObject : public transformational::Object {
private:
    implementations::SkyboxObject m_object;

public:
    interfaces::Object* object() override { return dynamic_cast<interfaces::Object*>(&m_object); }

    SkyboxObject(const utils::vkDefault::Paths& texturePaths);

    SkyboxObject(const SkyboxObject&) = delete;
    SkyboxObject& operator=(const SkyboxObject&) = delete;

    SkyboxObject(SkyboxObject&&) = delete;
    SkyboxObject& operator=(SkyboxObject&&) = delete;

    SkyboxObject& setColor(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);
};

} // moon::entities

#endif // MOON_ENTITIES_SKYBOX_OBJECT_H