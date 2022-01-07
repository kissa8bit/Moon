#ifndef MOON_DEFERRED_GRAPHICS_DEFERREDGRAPHICS_H
#define MOON_DEFERRED_GRAPHICS_DEFERREDGRAPHICS_H

#include <graphicsManager/graphicsInterface.h>

#include <math/linearAlgebra.h>

#include <workflows/workflow.h>
#include <interfaces/camera.h>
#include <interfaces/object.h>
#include <interfaces/light.h>
#include <interfaces/model.h>

#include <workflows/skybox.h>
#include <workflows/scattering.h>
#include <workflows/sslr.h>
#include <workflows/bloom.h>
#include <workflows/blur.h>
#include <workflows/boundingBox.h>
#include <workflows/ssao.h>
#include <workflows/selector.h>
#include <workflows/shadow.h>
#include <workflows/postProcessing.h>

#include <utils/pipelineNode.h>
#include <utils/cursor.h>
#include <utils/buffer.h>

#include "renderStages/graphics.h"
#include "renderStages/layersCombiner.h"
#include "renderStages/deferredLink.h"

namespace moon::deferredGraphics {

template <typename T>
class UpdateTracked {
private:
    T v;
    bool update{false};

public:
    UpdateTracked(const T& v) : v(v), update(true) {}
    UpdateTracked(const UpdateTracked& other) {
        v = other.v;
        update = true;
    }
    UpdateTracked& operator=(const UpdateTracked& other) {
        v = other.v;
        update = true;
        return *this;
    }
    operator T() { return v; }
    T release() { update = false; return v; }
    bool updated() const { return update; }
};

struct Parameters {
    std::filesystem::path       shadersPath;
    std::filesystem::path       workflowsShadersPath;
    math::vec2u                 extent{0};
    VkSampleCountFlagBits       MSAASamples{ VK_SAMPLE_COUNT_1_BIT };
    utils::Cursor*              cursor{ nullptr };
    interfaces::Camera*         cameraObject{ nullptr };

    struct {
        UpdateTracked<uint32_t> blitAttachmentsCount{ 8 };
        UpdateTracked<uint32_t> transparentLayersCount{ 2 };
        UpdateTracked<float>    blitFactor{ 1.5f };
        UpdateTracked<float>    blurDepth{ 1.0f };
        UpdateTracked<float>    minAmbientFactor{ 0.05f };
        UpdateTracked<bool>     scatteringRefraction{ false };
    } workflowsParameters;

    UpdateTracked<uint32_t>& blitAttachmentsCount() {return workflowsParameters.blitAttachmentsCount;}
    UpdateTracked<uint32_t>& transparentLayersCount() { return workflowsParameters.transparentLayersCount; }
    UpdateTracked<float>&    blitFactor() { return workflowsParameters.blitFactor; }
    UpdateTracked<float>&    blurDepth() { return workflowsParameters.blurDepth; }
    UpdateTracked<float>&    minAmbientFactor() { return workflowsParameters.minAmbientFactor; }
    UpdateTracked<bool>&     scatteringRefraction() { return workflowsParameters.scatteringRefraction; }
};

class DeferredGraphics: public graphicsManager::GraphicsInterface {
private:
    Parameters params;

    workflows::WorkflowsMap workflows;
    workflows::ParametersMap workflowsParameters;

    utils::vkDefault::CommandPool commandPool;
    utils::vkDefault::CommandBuffers copyCommandBuffers;
    utils::PipelineNodes nodes;

    utils::BuffersDatabase bDatabase;
    utils::AttachmentsDatabase aDatabase;

    interfaces::Objects objects;
    interfaces::Lights lights;
    interfaces::DepthMaps depthMaps;
    utils::TextureMap emptyTextures;

    GraphicsParameters graphicsParams;
    std::vector<GraphicsParameters> transparentLayersParams;
    LayersCombinerParameters layersCombinerParams;
    workflows::SkyboxParameters skyboxParams;
    workflows::ScatteringParameters scatteringParams;
    workflows::SSLRParameters SSLRParams;
    workflows::BloomParameters bloomParams;
    workflows::GaussianBlurParameters blurParams;
    workflows::BoundingBoxParameters bbParams;
    workflows::SSAOParameters SSAOParams;
    workflows::SelectorParameters selectorParams;
    workflows::PostProcessingParameters postProcessingParams;
    workflows::ShadowGraphicsParameters shadowGraphicsParameters;

    void createGraphicsPasses();
    void createStages();
    void updateParameters();

    void update(uint32_t imageIndex) override;
    utils::vkDefault::VkSemaphores submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;

public:
    DeferredGraphics(const Parameters& parameters);

    void reset() override;

    bool getEnable(const std::string& name);
    DeferredGraphics& setEnable(const std::string& name, bool enable);
    DeferredGraphics& requestUpdate(const std::string& name);

    Parameters& parameters();

    void create(interfaces::Model* pModel);

    void bind(interfaces::Camera* cameraObject);
    void remove(interfaces::Camera* cameraObject);

    void bind(interfaces::Object* object);
    bool remove(interfaces::Object* object);

    void bind(interfaces::Light* lightSource);
    bool remove(interfaces::Light* lightSource);

    void bind(utils::Cursor* cursor);
    bool remove(utils::Cursor* cursor);
};

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_DEFERREDGRAPHICS_H
