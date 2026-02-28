#include "skyboxObject.h"

namespace moon::entities {

SkyboxObject::SkyboxObject(const utils::vkDefault::Paths& texturePaths)
    : transformational::Object(), m_object(texturePaths)
{}

SkyboxObject& SkyboxObject::setColor(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    if (constant.has_value()) {
        m_object.buffer(true).base.constant = constant.value();
    }
    if (factor.has_value()) {
        m_object.buffer(true).base.factor = factor.value();
    }
    return *this;
}

}
