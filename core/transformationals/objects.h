#ifndef BASEOBJECT_H
#define BASEOBJECT_H

#include <vulkan.h>

#include <filesystem>
#include <optional>

#include "transformational.h"
#include "texture.h"
#include "buffer.h"
#include "object.h"
#include "model.h"

#include "linearAlgebra.h"

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
            alignas(16) math::vec4 constant{0.0f};
            alignas(16) math::vec4 factor{1.0f};
        };
        struct Outlining {
            alignas(16) math::vec4 color{ 0.0f };
            alignas(4) float width{ 0.0f };
        };

        alignas(16) math::mat4 modelMatrix;
        ColorLinearProperties base;
        ColorLinearProperties bloom;
        Outlining outlining;
    } buffer;

    DEFAULT_TRANSFORMATIONAL()

    std::unique_ptr<interfaces::Object> pObject;

public:
    Object() = default;
    Object(interfaces::Model* model, uint32_t firstInstance = 0, uint32_t instanceCount = 1);
    Object(const utils::Paths& texturePaths, const float& mipLevel = 1.0f);

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Object)
    DEFAULT_TRANSFORMATIONAL_GETTERS()

    Object& setOutlining(const bool& enable, const float& width = 0, const math::vec4& color = { 0.0f });
    Object& setBase(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);
    Object& setBloom(std::optional<math::vec4> constant = std::nullopt, std::optional<math::vec4> factor = std::nullopt);

    class AnimationControl {
    private:
        size_t total{0};
        std::map<size_t, std::vector<interfaces::Animation*>> animationsMap;
        float time{0};
        float startOffset{ 0 };
        int animIndex{ -1 };

        friend class Object;

    public:
        size_t size() const;
        size_t current() const;
        void set(int animIndex, float changeTime = 0);
        bool update(size_t frameNumber, float dtime);
    } animationControl;

    operator interfaces::Object* () const;
};

}
#endif // BASEOBJECT_H
