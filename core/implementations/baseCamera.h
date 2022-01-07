#ifndef MOON_IMPLEMENTATIONS_BASE_CAMERA_H
#define MOON_IMPLEMENTATIONS_BASE_CAMERA_H

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>

#include <math/linearAlgebra.h>

#include <interfaces/camera.h>

namespace moon::implementations {

class BaseCamera : public interfaces::Camera
{
public:
    struct Buffer {
        alignas(16) math::mat4 view{ 1.0f };
        alignas(16) math::mat4 proj{ 1.0f };
    };

    BaseCamera();
    Buffer& buffer(bool update);

    void setTransformation(const math::mat4& transformation) override;

private:
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;

    void create(const utils::PhysicalDevice& device, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    utils::Buffers& buffers() override;
};

} // moon::implementations

#endif // MOON_IMPLEMENTATIONS_BASE_CAMERA_H