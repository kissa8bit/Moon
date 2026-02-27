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
#include <utils/types.h>

#include "renderStages/graphics.h"
#include "renderStages/layersCombiner.h"
#include "renderStages/deferredLink.h"

namespace moon::deferredGraphics {

struct Names {
    static inline const utils::BufferName camera{ "camera" };
    static inline const utils::ImageName whiteTexture{ "white" };
    static inline const utils::ImageName blackTexture{ "black" };

    struct MainGraphics {
        static inline const workflows::WorkflowName name{ "MainGraphics" };
        static inline const workflows::ParameterName param{ "MainGraphics" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName image{ prefix + "image" };
        static inline const utils::AttachmentName bloom{ prefix + "bloom" };
        struct GBuffer {
            static inline const utils::AttachmentName prefix{ MainGraphics::prefix + "GBuffer." };
            static inline const utils::AttachmentName position{ prefix + "position" };
            static inline const utils::AttachmentName normal{ prefix + "normal" };
            static inline const utils::AttachmentName color{ prefix + "color" };
            static inline const utils::AttachmentName emission{ prefix + "emission" };
            static inline const utils::AttachmentName depth{ prefix + "depth" };
        };
    };

    struct LayersCombiner {
        static inline const workflows::WorkflowName name{ "LayersCombiner" };
        static inline const workflows::ParameterName param{ "LayersCombiner" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName color{ prefix + "color" };
        static inline const utils::AttachmentName bloom{ prefix + "bloom" };
        static inline const utils::AttachmentName blur{ prefix + "blur" };
    };

    struct PostProcessing {
        static inline const workflows::WorkflowName name{ "PostProcessing" };
        static inline const workflows::ParameterName param{ "PostProcessing" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct Bloom {
        static inline const workflows::WorkflowName name{ "Bloom" };
        static inline const workflows::ParameterName param{ "Bloom" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct Blur {
        static inline const workflows::WorkflowName name{ "Blur" };
        static inline const workflows::ParameterName param{ "Blur" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct Skybox {
        static inline const workflows::WorkflowName name{ "Skybox" };
        static inline const workflows::ParameterName param{ "Skybox" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName color{ prefix + "color" };
        static inline const utils::AttachmentName bloom{ prefix + "bloom" };
    };

    struct SSLR {
        static inline const workflows::WorkflowName name{ "SSLR" };
        static inline const workflows::ParameterName param{ "SSLR" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct SSAO {
        static inline const workflows::WorkflowName name{ "SSAO" };
        static inline const workflows::ParameterName param{ "SSAO" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct Shadow {
        static inline const workflows::WorkflowName name{ "Shadow" };
        static inline const workflows::ParameterName param{ "Shadow" };
    };

    struct Scattering {
        static inline const workflows::WorkflowName name{ "Scattering" };
        static inline const workflows::ParameterName param{ "Scattering" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct BoundingBox {
        static inline const workflows::WorkflowName name{ "BoundingBox" };
        static inline const workflows::ParameterName param{ "BoundingBox" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };

    struct Selector {
        static inline const workflows::WorkflowName name{ "Selector" };
        static inline const workflows::ParameterName param{ "Selector" };
        static inline const utils::AttachmentName prefix{ name + "." };
        static inline const utils::AttachmentName output{ prefix + "output" };
    };
};

struct Parameters {
    std::filesystem::path       shadersPath;
    std::filesystem::path       workflowsShadersPath;
    math::vec2u                 extent{0};
    VkSampleCountFlagBits       MSAASamples{ VK_SAMPLE_COUNT_1_BIT };
    math::vec2u                 shadowsExtent{ 1024, 1024 };
    utils::Cursor*              cursor{ nullptr };
    interfaces::Camera*         cameraObject{ nullptr };

    struct {
        utils::DirtyValue<uint32_t> blitAttachmentsCount{ 8 };
        utils::DirtyValue<uint32_t> layersCount{ 3 };
        utils::DirtyValue<float>    blitFactor{ 1.5f };
        utils::DirtyValue<float>    blurDepth{ 0.0f };
        utils::DirtyValue<float>    minAmbientFactor{ 0.0f };
    } workflowsParameters;

    utils::DirtyValue<uint32_t>& blitAttachmentsCount() {return workflowsParameters.blitAttachmentsCount;}
    utils::DirtyValue<uint32_t>& layersCount() { return workflowsParameters.layersCount; }
    utils::DirtyValue<float>&    blitFactor() { return workflowsParameters.blitFactor; }
    utils::DirtyValue<float>&    blurDepth() { return workflowsParameters.blurDepth; }
    utils::DirtyValue<float>&    minAmbientFactor() { return workflowsParameters.minAmbientFactor; }
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

    DeferredLink linkMember;

    void createGraphicsPasses();
    void createStages();
    void updateParameters();

    void update(uint32_t imageIndex) override;
    utils::vkDefault::VkSemaphores submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore = {}) override;
    void draw(VkCommandBuffer commandBuffer, uint32_t imageNumber) const override;

public:
    DeferredGraphics(const Parameters& parameters);

    void reset() override;

    bool getEnable(const workflows::ParameterName& name);
    DeferredGraphics& setEnable(const workflows::ParameterName& name, bool enable);
    DeferredGraphics& requestUpdate(const workflows::WorkflowName& name);

    Parameters& parameters();

    void create(interfaces::Model* pModel);

    void bind(interfaces::Camera* cameraObject);
    bool remove(interfaces::Camera* cameraObject);

    void bind(interfaces::Object* object);
    bool remove(interfaces::Object* object);

    void bind(interfaces::Light* lightSource);
    bool remove(interfaces::Light* lightSource);

    void bind(utils::Cursor* cursor);
    bool remove(utils::Cursor* cursor);
};

} // moon::deferredGraphics

#endif // MOON_DEFERRED_GRAPHICS_DEFERREDGRAPHICS_H
