#ifndef MOON_IMPLEMENTATIONS_POINT_LIGHT_H
#define MOON_IMPLEMENTATIONS_POINT_LIGHT_H

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>
#include <utils/texture.h>

#include <math/linearAlgebra.h>

#include <interfaces/light.h>

namespace moon::implementations {

class PointLight : public interfaces::Light {
public:
    struct Buffer {
        alignas(16) math::vec4 position;
        alignas(16) math::vec4 color;
        alignas(16) struct {
            float radius{ 10.0f };
            float powerFactor{ 10.0f };
            float dropFactor{ 1.0f };
            float _pad{ 0.0f };
        } props;
    };

    PointLight();
    Buffer& buffer(bool markDirty);
    const Buffer& buffer() const;

    void setTransformation(const math::mat4& transformation) override;

private:
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;
    utils::Texture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);
    utils::Buffers& buffers() override;
};

} // moon::implementations

#endif // MOON_IMPLEMENTATIONS_POINT_LIGHT_H
