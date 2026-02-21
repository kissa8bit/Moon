#ifndef MOON_IMPLEMENTATIONS_BASE_OBJECT_H
#define MOON_IMPLEMENTATIONS_BASE_OBJECT_H

#include <utils/vkdefault.h>
#include <utils/device.h>
#include <utils/buffer.h>

#include <math/linearAlgebra.h>

#include <interfaces/object.h>

namespace moon::implementations {

class BaseObject : public interfaces::Object {
public:
    struct Buffer{
        struct ColorLinearProperties {
            alignas(16) math::vec4 constant{ 0.0f };
            alignas(16) math::vec4 factor{ 1.0f };
        };

        struct Outlining {
            alignas(16) math::vec4 color{ 0.0f };
            alignas(4) float width{ 0.0f };
        };

        alignas(16) math::mat4 modelMatrix;
        ColorLinearProperties base;
        ColorLinearProperties bloom;
        Outlining outlining;
    };

    BaseObject(interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount);
    Buffer& buffer(bool markDirty);

    void setTransformation(const math::mat4& transformation) override;

    static utils::vkDefault::DescriptorSetLayout createDescriptorSetLayout(VkDevice device);

private:
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);
    utils::Buffers& buffers() override;
};

} // moon::implementations

#endif // MOON_IMPLEMENTATIONS_BASE_OBJECT_H
