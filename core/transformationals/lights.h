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

class SpotLight : public interfaces::Light
{
public:
    enum Type {
        circle,
        square
    };

    struct Props
    {
        bool enableShadow;
        bool enableScattering;
        operator interfaces::LightProperty() const;
    };

private:
    utils::UniformBuffer uniformBuffer;

    const std::filesystem::path texturePath;
    utils::Texture texture;

    void create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) override;
    void update(uint32_t frameNumber, VkCommandBuffer commandBuffer) override;
    void render(uint32_t frameNumber, VkCommandBuffer commandBuffer, const utils::vkDefault::DescriptorSets& descriptorSet, VkPipelineLayout pipelineLayout, VkPipeline pipeline) override;
    void createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount);

public:
    SpotLight(void* hostData, size_t hostDataSize, const Props& props, const std::filesystem::path& texturePaths = "");
    utils::Buffers& buffers() override;
};

}

namespace moon::transformational {

class Light : public Transformational
{
private:
    struct {
        alignas(16) math::mat4 proj;
        alignas(16) math::mat4 view;
        alignas(16) math::vec4 color;
        alignas(16) math::vec4 props;
    } buffer;

    DEFAULT_TRANSFORMATIONAL()

    std::unique_ptr<interfaces::Light> pLight;

    interfaces::SpotLight::Type type{ interfaces::SpotLight::Type::circle };
    float lightPowerFactor{ 10.0f };
    float lightDropFactor{ 1.0f };

public:
    Light(const math::vec4& color, const math::mat4& projection, bool enableShadow = true, bool enableScattering = false, interfaces::SpotLight::Type type = interfaces::SpotLight::Type::circle);
    Light(const std::filesystem::path& texturePath, const math::mat4& projection, bool enableShadow = true, bool enableScattering = false, interfaces::SpotLight::Type type = interfaces::SpotLight::Type::circle);

    DEFAULT_TRANSFORMATIONAL_OVERRIDE(Light)
    DEFAULT_TRANSFORMATIONAL_GETTERS()
    DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Light)

    Light& setColor(const math::vec4& color);
    Light& setDrop(const float& drop);
    Light& setPower(const float& power);
    Light& setProjectionMatrix(const math::mat4& projection);

    operator interfaces::Light* () const;
};

class IsotropicLight: public Group
{
private:
    std::vector<std::unique_ptr<Light>> lights;

public:
    IsotropicLight(const math::vec4& color = {0.0f}, float radius = 100.0f, bool enableShadow = true, bool enableScattering = false);
    ~IsotropicLight() = default;

    IsotropicLight& setColor(const math::vec4& color);
    IsotropicLight& setDrop(const float& drop);
    IsotropicLight& setPower(const float& power);
    IsotropicLight& setProjectionMatrix(const math::mat4& projection);

    std::vector<interfaces::Light*> getLights() const;
};

}
#endif // SPOTLIGHT_H
