#include "baseObject.h"

#include <implementations/baseObject.h>

namespace moon::entities {

BaseObject::BaseObject(interfaces::Model* model, uint32_t firstInstance , uint32_t instanceCount)
    : transformational::Object()
{
    pObject = std::make_unique<implementations::BaseObject>(model, firstInstance, instanceCount);

    pObject->objectMask().set(interfaces::ObjectProperty::enable, true);
    pObject->objectMask().set(interfaces::ObjectProperty::enableShadow, true);

    if (model) {
        for (auto instance = 0; instance < instanceCount; ++instance) {
            animationControl.animationsMap[instance] = model->animations(firstInstance + instance);
            animationControl.total = animationControl.animationsMap[instance].size();
        }
    }
}

bool BaseObject::isEnable() const {
    auto pBaseObject = static_cast<implementations::BaseObject*>(pObject.get());
    if (!pBaseObject) return false;

    return pBaseObject->objectMask().property().has(interfaces::ObjectProperty::enable);
}

BaseObject& BaseObject::setEnable(const bool enable) {
    auto pBaseObject = static_cast<implementations::BaseObject*>(pObject.get());
    if (!pBaseObject) return *this;

    pBaseObject->objectMask().property().set(interfaces::ObjectProperty::enable, enable);
    return *this;
}

BaseObject& BaseObject::setColor(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    auto pBaseObject = static_cast<implementations::BaseObject*>(pObject.get());
    if (!pBaseObject) return *this;

    if (pObject && constant.has_value()) {
        pBaseObject->buffer(true).base.constant = constant.value();
    }
    if (factor.has_value()) {
        pBaseObject->buffer(true).base.factor = factor.value();
    }

    return *this;
}

BaseObject& BaseObject::setBloom(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    auto pBaseObject = static_cast<implementations::BaseObject*>(pObject.get());
    if (!pBaseObject) return *this;

    if (constant.has_value()) {
        pBaseObject->buffer(true).bloom.constant = constant.value();
    }
    if (factor.has_value()) {
        pBaseObject->buffer(true).bloom.factor = factor.value();
    }

    return *this;
}

BaseObject& BaseObject::setOutlining(const bool enable, const float width, const math::vec4& color) {
    auto pBaseObject = static_cast<implementations::BaseObject*>(pObject.get());
    if (!pBaseObject) return *this;

    pBaseObject->objectMask().set(interfaces::ObjectType::outlining, enable);
    if (width > 0.0f) {
        pBaseObject->buffer(true).outlining.width = width;
    }
    if (math::dot(color, color) > 0.0f) {
        pBaseObject->buffer(true).outlining.color = color;
    }
    return *this;
}

size_t BaseObject::AnimationControl::size() const {
    return total;
}

size_t BaseObject::AnimationControl::current() const {
    return animIndex;
}

void BaseObject::AnimationControl::set(int index, float changeTime) {
    for (auto& [_, animations] : animationsMap) {
        if (index < static_cast<int>(animations.size())) {
            animIndex = index;
            startOffset = changeTime;
            time = 0;
            if (animIndex > -1) {
                animations.at(animIndex)->setChangeTime(changeTime);
            }
        }
    }
}

bool BaseObject::AnimationControl::update(size_t frameNumber, float dtime) {
    if (auto animationsIt = animationsMap.find(frameNumber); animationsIt != animationsMap.end()) {
        auto& [_, animations] = *animationsIt;

        if (animations.size() == 0 || animIndex < 0) return false;

        auto animation = animations.at(animIndex);
        if (!animation) return false;

        time += dtime;
        if (time > animation->duration()) {
            time = startOffset;
        }
        return animation->update(time);
    }
    return false;
}

} // moon::entities