#ifndef MOON_RAY_TRACING_GRAPHICS_RAYTRACINGGRAPHICS
#define MOON_RAY_TRACING_GRAPHICS_RAYTRACINGGRAPHICS

#include <stdint.h>
#include <random>
#include <stack>
#include <memory>

#include <utils/attachments.h>

#include <graphicsManager/graphicsInterface.h>

#include <workflows/bloom.h>

#include <math/linearAlgebra.h>

#include "boundingBoxGraphics.h"
#include "rayTracingLink.h"

#include <cudaRayTracing/graphics/cudaRayTracing.h>

namespace cuda::rayTracing { struct Object;}

namespace moon::rayTracingGraphics {

class RayTracingGraphics : public moon::graphicsManager::GraphicsInterface {
private:
    struct ImageResource{
        std::string id;
        std::vector<uint32_t> host;
        moon::utils::Buffer hostDevice;
        moon::utils::Attachments device;

        ImageResource() = default;
        ImageResource(const std::string& id, const moon::utils::PhysicalDevice& phDevice, const moon::utils::vkDefault::ImageInfo& imageInfo);
        ImageResource(const ImageResource&) = delete;
        ImageResource& operator=(const ImageResource&) = delete;
        ImageResource(ImageResource&&) noexcept;
        ImageResource& operator=(ImageResource&&) noexcept;
        void swap(ImageResource&) noexcept;

        void moveFromHostToHostDevice(VkExtent2D extent);
        void copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex);
    };

    ImageResource color;
    ImageResource bloom;

    cuda::rayTracing::RayTracing rayTracer;
    BoundingBoxGraphics bbGraphics;
    std::unique_ptr<moon::workflows::BloomGraphics> bloomGraph;

    moon::utils::Texture emptyTexture;

    std::filesystem::path shadersPath;
    std::filesystem::path workflowsShadersPath;
    VkExtent2D extent;

    moon::utils::AttachmentsDatabase    aDatabase;
    moon::utils::BuffersDatabase        bDatabase;

    utils::vkDefault::CommandPool commandPool;

    bool bloomEnable = true;

    moon::workflows::BloomParameters bloomParams;

    void update(uint32_t imageIndex) override;
    utils::vkDefault::VkSemaphores submit(uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;

public:
    RayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent);

    void reset() override;

    void setEnableBoundingBox(bool enable);
    void setEnableBloom(bool enable);
    void setBlitFactor(const float& blitFactor);
    void setExtent(VkExtent2D extent);

    void setCamera(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* cam);
    void bind(cuda::rayTracing::Object* obj);

    void clearFrame();
    void buildTree();
    void buildBoundingBoxes(bool primitive, bool tree, bool onlyLeafs);
};

} // moon::rayTracingGraphics

#endif // MOON_RAY_TRACING_GRAPHICS_RAYTRACINGGRAPHICS

