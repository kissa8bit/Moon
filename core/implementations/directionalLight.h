#ifndef MOON_IMPLEMENTATIONS_DIRECTIONAL_LIGHT_H
#define MOON_IMPLEMENTATIONS_DIRECTIONAL_LIGHT_H

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>
#include <utils/texture.h>

#include <math/linearAlgebra.h>

#include <interfaces/light.h>

namespace moon::implementations {

class DirectionalLight : public interfaces::Light {
public:
    struct Buffer {
        alignas(16) math::mat4 proj;
        alignas(16) math::mat4 view;
        alignas(16) math::vec4 color;
        alignas(16) struct {
            float pad0{ 0.0f };
            float pad1{ 0.0f };
            float powerFactor{ 10.0f };
            float dropFactor{ 0.0f };
        } props;
    };

    DirectionalLight();
    Buffer& buffer(bool markDirty);
    const Buffer& buffer() const;

    void setEnable(bool enable);
    bool isEnable() const;

    void setTransformation(const math::mat4& transformation) override;

private:
    bool enabled{ true };
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;
    utils::Texture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) override;
    void render(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);
    utils::Buffers& buffers() override;
};

} // moon::implementations

#endif // MOON_IMPLEMENTATIONS_DIRECTIONAL_LIGHT_H
