#ifndef MOON_ENTITIES_BASE_OBJECT_H
#define MOON_ENTITIES_BASE_OBJECT_H

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <transformationals/objects.h>
#include <implementations/baseObject.h>

namespace moon::entities {

class BaseObject : public transformational::Object {
private:
    implementations::BaseObject m_object;

public:
    interfaces::Object* object() override { return dynamic_cast<interfaces::Object*>(&m_object); }

    struct AnimationConfig {
        int index{ -1 };
        float blendTime{ 0.0f };
        float speed{ 1.0f };
        enum class Transition { Smooth, Instant } transition{ Transition::Smooth };
    };

    BaseObject(interfaces::Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1, AnimationConfig animConfig = {});

    BaseObject(const BaseObject&) = delete;
    BaseObject& operator=(const BaseObject&) = delete;

    BaseObject(BaseObject&&) = delete;
    BaseObject& operator=(BaseObject&&) = delete;

    bool isEnable() const;

    BaseObject& setEnable(const bool enable);
    BaseObject& setOutlining(const bool enable, const float width = 0, const math::vec4& color = { 0.0f });
    BaseObject& setColor(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);
    BaseObject& setBloom(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);

    // Called by the rendering system each frame, not part of the user-facing animation API
    bool updateAnimation(size_t frameNumber, float dtime);

    struct Animation {
        // Query
        size_t count() const;
        int current(int layer = 0) const;
        std::string_view name(size_t index) const;

        // Control — all return Animation& for fluent chaining
        // play() operates on layer 0, playLayer() on explicit layer
        Animation& play(int index, std::optional<float> blendTime = std::nullopt);
        Animation& play(std::string_view animName, std::optional<float> blendTime = std::nullopt);
        Animation& playLayer(int layer, int index, std::optional<float> blendTime = std::nullopt);
        Animation& playLayer(int layer, std::string_view animName, std::optional<float> blendTime = std::nullopt);
        Animation& stop();
        Animation& stopLayer(int layer);
        Animation& pause();
        Animation& resume();
        Animation& setSpeed(float speed);

        AnimationConfig config;

    private:
        explicit Animation(BaseObject& owner) : m_owner(owner) {}
        Animation(const Animation&) = delete;
        Animation& operator=(const Animation&) = delete;

        struct Layer {
            int animIndex{-1};
            float time{0.0f};
            bool paused{false};
        };

        BaseObject& m_owner;
        std::map<size_t, std::vector<interfaces::Animation*>> m_animationsMap;
        std::vector<std::string> m_names;
        std::map<int, Layer> m_layers;
        bool m_paused{ false };

        bool update(size_t frameNumber, float dtime);

        friend class BaseObject;
    } animation;
};

} // moon::entities

#endif // MOON_ENTITIES_BASE_OBJECT_H
