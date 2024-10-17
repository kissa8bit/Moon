#ifndef BASECAMERA_H
#define BASECAMERA_H

#include <vulkan.h>

#include "camera.h"
#include "transformational.h"
#include "buffer.h"

#include "linearAlgebra.h"

namespace moon::interfaces {

class BaseCamera : public Camera
{
private:
    utils::UniformBuffer uniformBuffer;

    void create(const utils::PhysicalDevice& device, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;

public:
    BaseCamera(void* hostData, size_t hostDataSize);
    utils::Buffers& buffers() override;
};

}

namespace moon::transformational {

class Camera : public Transformational
{
private:
    struct {
        alignas(16) math::mat4 view{ 1.0f };
        alignas(16) math::mat4 proj{ 1.0f };
    } buffer;

    DEFAULT_TRANSFORMATIONAL()

    std::unique_ptr<interfaces::Camera> pCamera;

public:
    Camera();
    Camera(const float& angle, const float& aspect, const float& n, const float& f = std::numeric_limits<float>::max());

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Camera)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Camera)

    Camera& setProjMatrix(const math::Matrix<float,4,4> & proj);
    math::mat4 getProjMatrix() const;
    math::mat4 getViewMatrix() const;

    operator interfaces::Camera*() const;
};

}
#endif // BASECAMERA_H
