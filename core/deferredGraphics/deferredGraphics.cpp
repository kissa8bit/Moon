#include "deferredGraphics.h"

#include <cstring>

#include <utils/operations.h>
#include <utils/texture.h>
#include <utils/swapChain.h>

namespace moon::deferredGraphics {

DeferredGraphics::DeferredGraphics(const Parameters& parameters)
    : params(parameters)
{
    link = std::make_unique<DeferredLink>();

    transparentLayersParams.resize(params.transparentLayersCount());
    (workflowsParameters["DeferredGraphics"] = &graphicsParams)->enable = true;
    (workflowsParameters["LayersCombiner"] = &layersCombinerParams)->enable = true;
    (workflowsParameters["PostProcessing"] = &postProcessingParams)->enable = true;
    (workflowsParameters["Bloom"] = &bloomParams)->enable = false;
    (workflowsParameters["Blur"] = &blurParams)->enable = false;
    (workflowsParameters["Skybox"] = &skyboxParams)->enable = false;
    (workflowsParameters["SSLR"] = &SSLRParams)->enable = false;
    (workflowsParameters["SSAO"] = &SSAOParams)->enable = false;
    (workflowsParameters["Shadow"] = &shadowGraphicsParameters)->enable = false;
    (workflowsParameters["Scattering"] = &scatteringParams)->enable = false;
    (workflowsParameters["BoundingBox"] = &bbParams)->enable = false;
    (workflowsParameters["TransparentLayer"] = &transparentLayersParams.front())->enable = false;
    (workflowsParameters["Selector"] = &selectorParams)->enable = false;
}

void DeferredGraphics::reset()
{
    commandPool = utils::vkDefault::CommandPool(device->device());
    copyCommandBuffers = commandPool.allocateCommandBuffers(resourceCount);

    aDatabase.destroy();
    emptyTextures["black"] = utils::Texture::empty(*device, commandPool);
    emptyTextures["white"] = utils::Texture::empty(*device, commandPool, false);
    aDatabase.addEmptyTexture("black", &emptyTextures["black"]);
    aDatabase.addEmptyTexture("white", &emptyTextures["white"]);

    createGraphicsPasses();
    createStages();
}

void DeferredGraphics::createGraphicsPasses(){
    CHECK_M(commandPool,            std::string("[ DeferredGraphics::createGraphicsPasses ] VkCommandPool is VK_NULL_HANDLE"));
    CHECK_M(device,                 std::string("[ DeferredGraphics::createGraphicsPasses ] device is nullptr"));
    CHECK_M(params.cameraObject,    std::string("[ DeferredGraphics::createGraphicsPasses ] camera is nullptr"));

    VkExtent2D extent{ params.extent[0], params.extent[1] };
    utils::vkDefault::ImageInfo info{ resourceCount, swapChainKHR->info().Format, extent, params.MSAASamples };
    utils::vkDefault::ImageInfo scatterInfo{ resourceCount, VK_FORMAT_R32G32B32A32_SFLOAT, extent, params.MSAASamples };
    utils::vkDefault::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, params.MSAASamples };

    graphicsParams.in.camera = "camera";
    graphicsParams.out.image = "image";
    graphicsParams.out.blur = "blur";
    graphicsParams.out.bloom = "bloom";
    graphicsParams.out.position = "GBuffer.position";
    graphicsParams.out.normal = "GBuffer.normal";
    graphicsParams.out.color = "GBuffer.color";
    graphicsParams.out.depth = "GBuffer.depth";
    graphicsParams.out.transparency = "transparency";
    graphicsParams.enableTransparency = workflowsParameters["TransparentLayer"]->enable;
    graphicsParams.transparencyPass = false;
    graphicsParams.transparencyNumber = 0;
    graphicsParams.minAmbientFactor = params.minAmbientFactor();
    graphicsParams.imageInfo = info;
    graphicsParams.shadersPath = params.shadersPath;

    skyboxParams.in.camera = graphicsParams.in.camera;
    skyboxParams.out.baseColor = "skybox.color";
    skyboxParams.out.bloom = "skybox.bloom";
    skyboxParams.imageInfo = info;
    skyboxParams.shadersPath = params.workflowsShadersPath;

    scatteringParams.in.camera = graphicsParams.in.camera;
    scatteringParams.in.depth = graphicsParams.out.depth;
    scatteringParams.out.scattering = "scattering";
    scatteringParams.imageInfo = scatterInfo;
    scatteringParams.shadersPath = params.workflowsShadersPath;

    SSLRParams.in.camera = graphicsParams.in.camera;
    SSLRParams.in.position = graphicsParams.out.position;
    SSLRParams.in.normal = graphicsParams.out.normal;
    SSLRParams.in.color = graphicsParams.out.image;
    SSLRParams.in.depth = graphicsParams.out.depth;
    SSLRParams.in.firstTransparency = graphicsParams.out.transparency + "0";
    SSLRParams.in.defaultDepthTexture = "white";
    SSLRParams.out.sslr = "sslr";
    SSLRParams.imageInfo = info;
    SSLRParams.shadersPath = params.workflowsShadersPath;

    layersCombinerParams.in.camera = graphicsParams.in.camera;
    layersCombinerParams.in.color = graphicsParams.out.image;
    layersCombinerParams.in.bloom = graphicsParams.out.bloom;
    layersCombinerParams.in.position = graphicsParams.out.position;
    layersCombinerParams.in.normal = graphicsParams.out.normal;
    layersCombinerParams.in.depth = graphicsParams.out.depth;
    layersCombinerParams.in.skyboxColor = skyboxParams.out.baseColor;
    layersCombinerParams.in.skyboxBloom = skyboxParams.out.bloom;
    layersCombinerParams.in.scattering = scatteringParams.out.scattering;
    layersCombinerParams.in.sslr = SSLRParams.out.sslr;
    layersCombinerParams.in.transparency = graphicsParams.out.transparency;
    layersCombinerParams.in.defaultDepthTexture = "white";
    layersCombinerParams.out.color = "combined.color";
    layersCombinerParams.out.bloom = "combined.bloom";
    layersCombinerParams.out.blur = "combined.blur";
    layersCombinerParams.enableTransparentLayers = workflowsParameters["TransparentLayer"]->enable;
    layersCombinerParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? params.transparentLayersCount() : 1;
    layersCombinerParams.enableScatteringRefraction = params.scatteringRefraction();
    layersCombinerParams.blurDepth = params.blurDepth();
    layersCombinerParams.imageInfo = info;
    layersCombinerParams.shadersPath = params.shadersPath;

    bloomParams.in.bloom = layersCombinerParams.out.bloom;
    bloomParams.out.bloom = "bloomFinal";
    bloomParams.blitAttachmentsCount = params.blitAttachmentsCount();
    bloomParams.blitFactor = params.blitFactor();
    bloomParams.xSamplerStep = params.blitFactor();
    bloomParams.ySamplerStep = params.blitFactor();
    bloomParams.imageInfo = info;
    bloomParams.shadersPath = params.workflowsShadersPath;

    blurParams.in.blur = layersCombinerParams.out.blur;
    blurParams.out.blur = "blured";
    blurParams.blurDepth = params.blurDepth();
    blurParams.imageInfo = info;
    blurParams.shadersPath = params.workflowsShadersPath;

    bbParams.in.camera = graphicsParams.in.camera;
    bbParams.out.boundingBox = "boundingBox";
    bbParams.imageInfo = info;
    bbParams.shadersPath = params.workflowsShadersPath;

    SSAOParams.in.camera = graphicsParams.in.camera;
    SSAOParams.in.position = graphicsParams.out.position;
    SSAOParams.in.normal = graphicsParams.out.normal;
    SSAOParams.in.color = graphicsParams.out.image;
    SSAOParams.in.depth = graphicsParams.out.depth;
    SSAOParams.in.defaultDepthTexture = "white";
    SSAOParams.out.ssao = "ssao";
    SSAOParams.imageInfo = info;
    SSAOParams.shadersPath = params.workflowsShadersPath;

    selectorParams.in.storageBuffer = "storage";
    selectorParams.in.position = graphicsParams.out.position;
    selectorParams.in.depth = graphicsParams.out.depth;
    selectorParams.in.transparency = graphicsParams.out.transparency;
    selectorParams.in.defaultDepthTexture = "white";
    selectorParams.out.selector = "selector";
    selectorParams.transparentLayersCount = workflowsParameters["TransparentLayer"]->enable ? params.transparentLayersCount() : 1;
    selectorParams.imageInfo = info;
    selectorParams.shadersPath = params.workflowsShadersPath;

    postProcessingParams.in.baseColor = layersCombinerParams.out.color;
    postProcessingParams.in.bloom = bloomParams.out.bloom;
    postProcessingParams.in.blur = blurParams.out.blur;
    postProcessingParams.in.boundingBox = bbParams.out.boundingBox;
    postProcessingParams.in.ssao = SSAOParams.out.ssao;
    postProcessingParams.out.postProcessing = "final";
    postProcessingParams.imageInfo = info;
    postProcessingParams.shadersPath = params.workflowsShadersPath,

    shadowGraphicsParameters.imageInfo = shadowsInfo;
    shadowGraphicsParameters.shadersPath = params.workflowsShadersPath,

    workflows.clear();

    workflows["DeferredGraphics"] = std::make_unique<Graphics>(graphicsParams, &objects, &lights, &depthMaps);
    workflows["LayersCombiner"] = std::make_unique<LayersCombiner>(layersCombinerParams);
    workflows["PostProcessing"] = std::make_unique<workflows::PostProcessingGraphics>(postProcessingParams);

    for(uint32_t i = 0; i < params.transparentLayersCount(); i++){
        const auto key = "TransparentLayer" + std::to_string(i);
        transparentLayersParams[i].in = graphicsParams.in;
        transparentLayersParams[i].out = graphicsParams.out;
        transparentLayersParams[i].enable = workflowsParameters["TransparentLayer"]->enable;
        transparentLayersParams[i].enableTransparency = true;
        transparentLayersParams[i].transparencyPass = true;
        transparentLayersParams[i].transparencyNumber = i;
        transparentLayersParams[i].minAmbientFactor = params.minAmbientFactor();
        transparentLayersParams[i].imageInfo = graphicsParams.imageInfo;
        transparentLayersParams[i].shadersPath = graphicsParams.shadersPath;
        workflows[key] = std::make_unique<Graphics>(transparentLayersParams[i], &objects, &lights, &depthMaps);
    };

    workflows["Blur"] = std::make_unique<workflows::GaussianBlur>(blurParams);
    workflows["Bloom"] = std::make_unique<workflows::BloomGraphics>(bloomParams);
    workflows["Skybox"] = std::make_unique<workflows::SkyboxGraphics>(skyboxParams, &objects);
    workflows["SSLR"] = std::make_unique<workflows::SSLRGraphics>(SSLRParams);
    workflows["SSAO"] = std::make_unique<workflows::SSAOGraphics>(SSAOParams);
    workflows["Shadow"] = std::make_unique<workflows::ShadowGraphics>(shadowGraphicsParameters, &objects, &depthMaps);
    workflows["Scattering"] = std::make_unique<workflows::Scattering>(scatteringParams, &lights, &depthMaps);
    workflows["BoundingBox"] = std::make_unique<workflows::BoundingBoxGraphics>(bbParams, &objects);
    workflows["Selector"] = std::make_unique<workflows::SelectorGraphics>(selectorParams, &params.cursor);

    for(auto& [k,workflow]: workflows){
        workflow->setDeviceProp(*device, device->device());
        workflow->create(commandPool, aDatabase);
    }

    for (auto& [_, workflow] : workflows) {
        workflow->updateDescriptors(bDatabase, aDatabase);
    }

    utils::vkDefault::ImageInfo linkInfo{resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, params.MSAASamples};
    link = std::make_unique<DeferredLink>(device->device(), params.shadersPath, linkInfo, link->renderPass(), link->positionInWindow(), aDatabase.get(postProcessingParams.out.postProcessing));
}

void DeferredGraphics::createStages(){
    nodes.clear();
    nodes.reserve(6);

    utils::PipelineStages postProcessingStages;
    postProcessingStages.push_back(
        utils::PipelineStage({*workflows["Selector"], *workflows["SSAO"], *workflows["Bloom"], *workflows["Blur"], *workflows["BoundingBox"], *workflows["PostProcessing"]},
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(postProcessingStages), nullptr));

    utils::PipelineStages layersCombinerStages;
    layersCombinerStages.push_back(utils::PipelineStage({ *workflows["LayersCombiner"] }, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(layersCombinerStages), &nodes.back()));

    utils::PipelineStages preCombinedStages;
    preCombinedStages.push_back(utils::PipelineStage({*workflows["Scattering"]}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    preCombinedStages.push_back(utils::PipelineStage({*workflows["SSLR"]}, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(preCombinedStages), &nodes.back()));

    std::vector<const utils::vkDefault::CommandBuffers*> deferredStagesCommandBuffers;
    deferredStagesCommandBuffers.push_back(*workflows["DeferredGraphics"]);
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        deferredStagesCommandBuffers.push_back(*workflows["TransparentLayer" + std::to_string(i)]);
    }
    utils::PipelineStages deferredStages;
    deferredStages.push_back(utils::PipelineStage(deferredStagesCommandBuffers, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(deferredStages), &nodes.back()));

    utils::PipelineStages prepareStages;
    prepareStages.push_back(utils::PipelineStage({*workflows["Shadow"]}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->device()(0,0)));
    prepareStages.push_back(utils::PipelineStage({*workflows["Skybox"]}, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(prepareStages), &nodes.back()));

    utils::PipelineStages copyStages;
    copyStages.push_back(utils::PipelineStage({ &copyCommandBuffers }, VK_PIPELINE_STAGE_TRANSFER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(copyStages), &nodes.back()));
}

utils::vkDefault::VkSemaphores DeferredGraphics::submit(const uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore) {
    return nodes.back().submit(frameIndex, externalSemaphore);
}

void DeferredGraphics::updateParameters() {
    if (params.blitFactor().updated()) {
        const auto blitFactor = params.blitFactor().release();
        bloomParams.blitFactor = blitFactor;
        bloomParams.xSamplerStep = blitFactor;
        bloomParams.ySamplerStep = blitFactor;
        workflows["Bloom"]->raiseUpdateFlags();
    }
    if (params.blurDepth().updated()) {
        const auto blurDepth = params.blurDepth().release();
        blurParams.blurDepth = blurDepth;
        layersCombinerParams.blurDepth = blurDepth;
        workflows["Blur"]->raiseUpdateFlags();
        workflows["LayersCombiner"]->raiseUpdateFlags();
    }
    if (params.minAmbientFactor().updated()) {
        const auto minAmbientFactor = params.minAmbientFactor().release();
        graphicsParams.minAmbientFactor = minAmbientFactor;
        workflows["DeferredGraphics"]->raiseUpdateFlags();
        for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
            transparentLayersParams[i].minAmbientFactor = minAmbientFactor;
            const auto key = "TransparentLayer" + std::to_string(i);
            workflows[key]->raiseUpdateFlags();
        };
    }
    if (params.scatteringRefraction().updated()) {
        layersCombinerParams.enableScatteringRefraction = params.scatteringRefraction().release();
        workflows["LayersCombiner"]->raiseUpdateFlags();
    }
}

void DeferredGraphics::update(uint32_t imageIndex) {
    updateParameters();

    CHECK(copyCommandBuffers[imageIndex].reset());
    CHECK(copyCommandBuffers[imageIndex].begin());
    if (params.cameraObject) {
        params.cameraObject->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& object : objects) {
        object->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    for (auto& light : lights) {
        light->update(imageIndex, copyCommandBuffers[imageIndex]);
    }
    CHECK(copyCommandBuffers[imageIndex].end());

    for (auto& [name, workflow] : workflows) {
        workflow->update(imageIndex);
    }
}

void DeferredGraphics::create(interfaces::Model *pModel){
    pModel->create(*device, commandPool);
}

void DeferredGraphics::bind(interfaces::Camera* cameraObject){
    params.cameraObject = cameraObject;
    cameraObject->create(*device, resourceCount);
    bDatabase.add("camera", &cameraObject->buffers());
}

void DeferredGraphics::remove(interfaces::Camera* cameraObject){
    if(params.cameraObject == cameraObject){
        params.cameraObject = nullptr;
        bDatabase.remove("camera");
    }
}

void DeferredGraphics::bind(interfaces::Light* lightSource){
    if(!lightSource) return;

    lightSource->create(*device, commandPool, resourceCount);
    lights.push_back(lightSource);

    if (depthMaps.find(lightSource) == depthMaps.end()) {
        utils::vkDefault::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, VkExtent2D{1024,1024}, params.MSAASamples };
        const auto lightProps = lightSource->lightMask().property();
        depthMaps[lightSource] = utils::DepthMap(*device, commandPool, shadowsInfo);
        depthMaps[lightSource].update(lightProps.has(interfaces::LightProperty::enableShadow) && workflowsParameters["Shadow"]->enable);
    }

    workflows["Shadow"]->raiseUpdateFlags();
    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Scattering"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
}

bool DeferredGraphics::remove(interfaces::Light* lightSource){
    size_t size = lights.size();
    lights.erase(std::remove(lights.begin(), lights.end(), lightSource), lights.end());

    if(depthMaps.count(lightSource)){
        depthMaps.erase(lightSource);
        workflows["Shadow"]->raiseUpdateFlags();
    }

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Scattering"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };

    return size - lights.size() > 0;
}

void DeferredGraphics::bind(interfaces::Object* object){
    object->create(*device, commandPool, resourceCount);
    objects.push_back(object);

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Skybox"]->raiseUpdateFlags();
    workflows["Shadow"]->raiseUpdateFlags();
    workflows["BoundingBox"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
}

bool DeferredGraphics::remove(interfaces::Object* object){
    size_t size = objects.size();
    objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());

    workflows["DeferredGraphics"]->raiseUpdateFlags();
    workflows["Skybox"]->raiseUpdateFlags();
    workflows["Shadow"]->raiseUpdateFlags();
    workflows["BoundingBox"]->raiseUpdateFlags();
    for (uint32_t i = 0; i < params.transparentLayersCount(); i++) {
        const auto key = "TransparentLayer" + std::to_string(i);
        workflows[key]->raiseUpdateFlags();
    };
    return size - objects.size() > 0;
}

void DeferredGraphics::bind(utils::Cursor* cursor) {
    params.cursor = cursor;
    cursor->create(*device, device->device());
    if (workflows["Selector"]) workflows["Selector"]->raiseUpdateFlags();
}

bool DeferredGraphics::remove(utils::Cursor* cursor) {
    return params.cursor == cursor ? !(params.cursor = nullptr) : false;
}

DeferredGraphics& DeferredGraphics::requestUpdate(const std::string& name) {
    workflows[name]->raiseUpdateFlags();
    return *this;
}

DeferredGraphics& DeferredGraphics::setEnable(const std::string& name, bool enable){
    workflowsParameters[name]->enable = enable;
    return *this;
}

bool DeferredGraphics::getEnable(const std::string& name){
    return workflowsParameters[name]->enable;
}

Parameters& DeferredGraphics::parameters() {
    return params;
}

} // moon::deferredGraphics
