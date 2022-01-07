#ifndef MOON_ENTITIES_BASE_OBJECT_H
#define MOON_ENTITIES_BASE_OBJECT_H

#include <optional>

#include <transformationals/objects.h>

namespace moon::entities {

class BaseObject : public transformational::Object {
public:
    BaseObject(interfaces::Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);

    BaseObject(const BaseObject&) = delete;
    BaseObject& operator=(const BaseObject&) = delete;

    BaseObject(BaseObject&&) = default;
    BaseObject& operator=(BaseObject&&) = default;

    bool isEnable() const;

    BaseObject& setEnable(const bool enable);
    BaseObject& setOutlining(const bool enable, const float width = 0, const math::vec4& color = { 0.0f });
    BaseObject& setColor(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);
    BaseObject& setBloom(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);

    class AnimationControl {
    private:
        size_t total{ 0 };
        std::map<size_t, std::vector<interfaces::Animation*>> animationsMap;
        float time{ 0 };
        float startOffset{ 0 };
        int animIndex{ -1 };

        friend class BaseObject;

    public:
        size_t size() const;
        size_t current() const;
        void set(int animIndex, float changeTime = 0);
        bool update(size_t frameNumber, float dtime);
    } animationControl;
};

} // moon::entities

#endif // MOON_ENTITIES_BASE_OBJECT_H