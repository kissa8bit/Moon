#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include <vulkan.h>

#include <filesystem>
#include <optional>

#include "transformational.h"
#include "quaternion.h"
#include "texture.h"
#include "buffer.h"
#include "object.h"
#include "matrix.h"
#include "model.h"

namespace moon::interfaces {

class BaseObject : public interfaces::Object {
protected:
    utils::UniformBuffer uniformBuffer;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);

public:
    virtual ~BaseObject() {};
    BaseObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize);
    BaseObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize, interfaces::Model* model, uint32_t firstInstance, uint32_t instanceCount);
    utils::Buffers& buffers() override;
};

class SkyboxObject : public BaseObject {
private:
    utils::Paths texturePaths;
    utils::CubeTexture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);

public:
    SkyboxObject(uint8_t pipelineBitMask, void* hostData, size_t hostDataSize, const utils::Paths& texturePaths, const float& mipLevel);
    SkyboxObject& setMipLevel(float mipLevel);
};

}

namespace moon::transformational {

class Object : public Transformational
{
private:
    struct {
        struct ColorLinearProperties {
            alignas(16) math::Vector<float, 4> constant{0.0f};
            alignas(16) math::Vector<float, 4> factor{1.0f};
        };
        alignas(16) math::Matrix<float, 4, 4> modelMatrix;
        ColorLinearProperties base;
        ColorLinearProperties bloom;
    } buffer;

    DEFAULT_TRANSFORMATIONAL()

    std::unique_ptr<interfaces::Object> pObject;

    bool changeAnimationFlag{false};
    uint32_t animationIndex{0};
    uint32_t newAnimationIndex{0};
    float animationTimer{0.0f};
    float startTimer{0.0f};
    float changeAnimationTime{0.0f};

public:
    Object() = default;
    Object(interfaces::Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    Object(const utils::Paths& texturePaths, const float& mipLevel = 1.0f);

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Object)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    Object& setOutlining(const bool& enable, const float& width = 0, const math::Vector<float, 4>& color = { 0.0f });
    Object& setBase(std::optional<math::Vector<float,4>> constant = std::nullopt, std::optional<math::Vector<float, 4>> factor = std::nullopt);
    Object& setBloom(std::optional<math::Vector<float, 4>> constant = std::nullopt, std::optional<math::Vector<float, 4>> factor = std::nullopt);

    uint32_t getAnimationIndex();
    void setAnimation(uint32_t animationIndex, float animationTime);
    void changeAnimation(uint32_t newAnimationIndex, float changeAnimationTime);
    void updateAnimation(uint32_t imageNumber, float frameTime);

    operator interfaces::Object* () const;
};

}
#endif // BASEOBJECT_H
