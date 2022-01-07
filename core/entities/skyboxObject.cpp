#include "skyboxObject.h"

#include <implementations/skyboxObject.h>

namespace moon::entities {

SkyboxObject::SkyboxObject(const utils::vkDefault::Paths& texturePaths, const float& mipLevel)
    : transformational::Object()
{
    pObject = std::make_unique<implementations::SkyboxObject>(texturePaths, mipLevel);
}

SkyboxObject& SkyboxObject::setColor(std::optional<math::vec4> constant, std::optional<math::vec4> factor) {
    auto pSkyboxObject = static_cast<implementations::SkyboxObject*>(pObject.get());
    if (!pSkyboxObject) return *this;

    if (constant.has_value()) {
        pSkyboxObject->buffer(true).base.constant = constant.value();
    }
    if (factor.has_value()) {
        pSkyboxObject->buffer(true).base.factor = factor.value();
    }
    return *this;
}

}
