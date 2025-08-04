#ifndef MOON_IMPLEMENTATIONS_SKYBOX_OBJECT_H
#define MOON_IMPLEMENTATIONS_SKYBOX_OBJECT_H

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>

#include <math/linearAlgebra.h>

#include <interfaces/object.h>

#include "baseObject.h"

namespace moon::implementations {

class SkyboxObject : public interfaces::Object {
public:
    struct Buffer {
        alignas(16) math::mat4 modelMatrix;
        BaseObject::Buffer::ColorLinearProperties base;
    };

    SkyboxObject(const utils::vkDefault::Paths& texturePaths, const float& mipLevel);
    Buffer& buffer(bool update);

    SkyboxObject& setMipLevel(float mipLevel);

    void setTransformation(const math::mat4& transformation) override;

    static utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);

private:
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;

    utils::vkDefault::Paths texturePaths;
    utils::CubeTexture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);
    utils::Buffers& buffers() override;
};


} // moon::implementations

#endif // MOON_IMPLEMENTATIONS_SKYBOX_OBJECT_H