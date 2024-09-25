#ifndef OBJECT_H
#define OBJECT_H

#include <vulkan.h>
#include <vector>
#include <vector.h>

#include "vkdefault.h"
#include "device.h"
#include "buffer.h"
#include "model.h"

namespace moon::interfaces {

enum ObjectType : uint8_t {
    base = 0x1,
    skybox = 0x2
};

enum ObjectProperty : uint8_t {
    non = 0x0,
    outlining = 1<<4
};

class Object {
protected:
    bool enable{true};
    bool enableShadow{true};

    uint32_t firstPrimitive{0};
    uint32_t primitiveCount{0};

    uint8_t pipelineBitMask{ 0 };
    Model* pModel{nullptr};
    uint32_t firstInstance{0};
    uint32_t instanceCount{1};

    utils::vkDefault::DescriptorSetLayout descriptorSetLayout;
    utils::vkDefault::DescriptorPool descriptorPool;
    utils::vkDefault::DescriptorSets descriptors;

public:
    Object() = default;
    Object(uint8_t pipelineBitMask);
    Object(uint8_t pipelineBitMask, Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);

    Model* model();
    uint32_t getInstanceNumber(uint32_t imageNumber) const;

    void setEnable(const bool& enable);
    void setEnableShadow(const bool& enable);
    bool getEnable() const;
    bool getEnableShadow() const;
    bool outlining() const;

    uint8_t& pipelineFlagBits();

    bool comparePrimitive(uint32_t primitive);
    void setFirstPrimitive(uint32_t firstPrimitive);
    void setPrimitiveCount(uint32_t primitiveCount);
    uint32_t getFirstPrimitive() const;
    uint32_t getPrimitiveCount() const;

    const VkDescriptorSet& getDescriptorSet(uint32_t i) const;

    virtual ~Object() {};
    virtual utils::Buffers& buffers() = 0;
    virtual void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) = 0;
    virtual void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) = 0;

    static utils::vkDefault::DescriptorSetLayout createBaseDescriptorSetLayout(VkDevice device);
    static utils::vkDefault::DescriptorSetLayout createSkyboxDescriptorSetLayout(VkDevice device);
};

using Objects = std::vector<interfaces::Object*>;

}
#endif // OBJECT_H
