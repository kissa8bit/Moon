#include "baseObject.h"

namespace moon::entities {

BaseObject::BaseObject(interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount, AnimationConfig animConfig)
    : transformational::Object(), animation(*this), m_object(model, firstInstance, instanceCount)
{
    m_object.objectMask().set(interfaces::ObjectProperty::enable, true);
    m_object.objectMask().set(interfaces::ObjectProperty::enableShadow, true);

    if (model) {
        for (auto instance = 0; instance < static_cast<int>(instanceCount); ++instance) {
            animation.m_animationsMap[instance] = model->animations(firstInstance + instance);
        }
        animation.m_names = model->animationNames();
    }

    animation.config = animConfig;
    if (animConfig.index >= 0) {
        animation.play(animConfig.index);
    }
}

bool BaseObject::isEnable() const {
    return m_object.objectMask().property().has(interfaces::ObjectProperty::enable);
}

BaseObject& BaseObject::setEnable(const bool enable) {
    m_object.objectMask().property().set(interfaces::ObjectProperty::enable, enable);
    return *this;
}

BaseObject& BaseObject::setColor(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    if (constant.has_value()) m_object.buffer(true).base.constant = constant.value();
    if (factor.has_value()) m_object.buffer(true).base.factor = factor.value();
    return *this;
}

BaseObject& BaseObject::setBloom(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    if (constant.has_value()) m_object.buffer(true).bloom.constant = constant.value();
    if (factor.has_value()) m_object.buffer(true).bloom.factor = factor.value();
    return *this;
}

BaseObject& BaseObject::setOutlining(const bool enable, const float width, const math::vec4& color) {
    m_object.objectMask().set(interfaces::ObjectType::outlining, enable);
    if (width > 0.0f) m_object.buffer(true).outlining.width = width;
    if (math::dot(color, color) > 0.0f) m_object.buffer(true).outlining.color = color;
    return *this;
}

bool BaseObject::updateAnimation(size_t frameNumber, float dtime) {
    return animation.update(frameNumber, dtime);
}

// --- Animation ---

size_t BaseObject::Animation::count() const {
    return m_animationsMap.empty() ? 0 : m_animationsMap.begin()->second.size();
}

int BaseObject::Animation::current() const {
    return m_animIndex;
}

std::string_view BaseObject::Animation::name(size_t index) const {
    return index < m_names.size() ? std::string_view(m_names[index]) : std::string_view{};
}

BaseObject::Animation& BaseObject::Animation::play(int index, std::optional<float> blendTime) {
    if (index < 0 || index >= static_cast<int>(count())) return *this;
    m_animIndex = index;
    m_time = 0.0f;
    m_paused = false;
    const bool instant = (config.transition == AnimationConfig::Transition::Instant);
    const float resolvedBlend = instant ? 0.0f : blendTime.value_or(config.blendTime);
    for (auto& [_, animations] : m_animationsMap) {
        animations.at(m_animIndex)->setChangeTime(resolvedBlend);
    }
    return *this;
}

BaseObject::Animation& BaseObject::Animation::play(std::string_view animName, std::optional<float> blendTime) {
    for (size_t i = 0; i < m_names.size(); ++i) {
        if (m_names[i] == animName) return play(static_cast<int>(i), blendTime);
    }
    return *this;
}

BaseObject::Animation& BaseObject::Animation::stop() {
    m_animIndex = -1;
    m_time = 0.0f;
    m_paused = false;
    return *this;
}

BaseObject::Animation& BaseObject::Animation::pause() {
    m_paused = true;
    return *this;
}

BaseObject::Animation& BaseObject::Animation::resume() {
    m_paused = false;
    return *this;
}

BaseObject::Animation& BaseObject::Animation::setSpeed(float speed) {
    config.speed = speed;
    return *this;
}

bool BaseObject::Animation::update(size_t frameNumber, float dtime) {
    if (m_paused || m_animIndex < 0) return false;

    auto it = m_animationsMap.find(frameNumber);
    if (it == m_animationsMap.end()) return false;

    auto& [_, animations] = *it;
    if (m_animIndex >= static_cast<int>(animations.size())) return false;

    auto* anim = animations.at(m_animIndex);
    if (!anim) return false;

    m_time += dtime * config.speed;
    if (m_time > anim->duration()) m_time = 0.0f;

    return anim->update(m_time);
}

} // moon::entities
