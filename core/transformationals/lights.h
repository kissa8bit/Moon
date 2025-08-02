#ifndef SPOTLIGHT_H
#define SPOTLIGHT_H

#include "transformational.h"
#include "light.h"
#include "buffer.h"
#include "texture.h"
#include "group.h"

#include "linearAlgebra.h"

#include <filesystem>
#include <memory>

namespace moon::interfaces {

class SpotLight : public interfaces::Light {
public:
    enum Type {
        circle,
        square
    };

    struct Buffer {
        alignas(16) math::mat4 proj;
        alignas(16) math::mat4 view;
        alignas(16) math::vec4 color;
        alignas(16) struct {
            float x{ 0.0 };
            float powerFactor{ 10.0f };
            float dropFactor{ 1.0f };
            float w{ 0.0f };
        }props;
    };

    SpotLight(Type type);
    Buffer& buffer(bool update);

    void setTexture(const std::filesystem::path& texturePath);
    void setTransformation(const math::mat4& transformation) override;

private:
    Type type{ Type::circle };
    std::filesystem::path texturePath;
    Buffer hostBuffer;
    utils::UniformBuffer uniformBuffer;
    utils::Texture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);
    utils::Buffers& buffers() override;
};

}

namespace moon::transformational {

class Light : public Transformational {
protected:
    std::unique_ptr<interfaces::Light> pLight;

    DEFAULT_TRANSFORMATIONAL()

public:
    Light(interfaces::LightType type);

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Light)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Light)

    operator interfaces::Light* () const;
};

}

namespace moon::entities {

class SpotLight : public transformational::Light {
public:
    struct Coloring
    {
        math::vec4 uniformColor{ 0.0f };
        std::filesystem::path texturePath{};

        Coloring(const math::vec4& uniformColor) : uniformColor(uniformColor) {}
        Coloring(const std::filesystem::path& texturePath) : texturePath(texturePath) {}
    };

    struct Props
    {
        bool enableShadow{ true };
        bool enableScattering{ false };
        float drop{ 1.0 };
        float power{ 10.0f };
    };

    SpotLight(const Coloring& coloring, const math::mat4& projection, const Props& props = {}, interfaces::SpotLight::Type type = interfaces::SpotLight::Type::circle);

    SpotLight& setColor(const math::vec4& color);
    SpotLight& setDrop(const float& drop);
    SpotLight& setPower(const float& power);
    SpotLight& setProjectionMatrix(const math::mat4& projection);
};

class IsotropicLight : public transformational::Group {
private:
    std::vector<SpotLight> lights;

public:
    IsotropicLight(const math::vec4& color = { 0.0f }, float radius = 100.0f, bool enableShadow = true, bool enableScattering = false);
    ~IsotropicLight() = default;

    IsotropicLight& setColor(const math::vec4& color);
    IsotropicLight& setDrop(const float& drop);
    IsotropicLight& setPower(const float& power);
    IsotropicLight& setProjectionMatrix(const math::mat4& projection);

    std::vector<interfaces::Light*> getLights() const;
};

}
#endif // SPOTLIGHT_H
