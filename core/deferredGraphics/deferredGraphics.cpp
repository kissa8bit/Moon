#include "deferredGraphics.h"

#include <cstring>

#include <utils/types.h>
#include <utils/operations.h>
#include <utils/texture.h>
#include <utils/swapChain.h>

namespace moon::deferredGraphics {

DeferredGraphics::DeferredGraphics(const Parameters& parameters)
    : params(parameters)
{
    (workflowsParameters[Names::MainGraphics::param] = &graphicsParams)->enable = true;
    (workflowsParameters[Names::LayersCombiner::param] = &layersCombinerParams)->enable = true;
    (workflowsParameters[Names::PostProcessing::param] = &postProcessingParams)->enable = true;
    (workflowsParameters[Names::Bloom::param] = &bloomParams)->enable = false;
    (workflowsParameters[Names::Blur::param] = &blurParams)->enable = false;
    (workflowsParameters[Names::Skybox::param] = &skyboxParams)->enable = false;
    (workflowsParameters[Names::SSLR::param] = &SSLRParams)->enable = false;
    (workflowsParameters[Names::SSAO::param] = &SSAOParams)->enable = false;
    (workflowsParameters[Names::Shadow::param] = &shadowGraphicsParameters)->enable = false;
    (workflowsParameters[Names::Scattering::param] = &scatteringParams)->enable = false;
    (workflowsParameters[Names::BoundingBox::param] = &bbParams)->enable = false;
    (workflowsParameters[Names::Selector::param] = &selectorParams)->enable = false;

    for (LayerIndex i{ 0 }; i < LayerIndex(params.layersCount); i++) {
        const auto key = layerPrefix<workflows::WorkflowName>(i) + Names::MainGraphics::name;
        params.workflowsToAllocate.insert(key);
    }
    params.workflowsToAllocate.insert(Names::LayersCombiner::name);
    params.workflowsToAllocate.insert(Names::PostProcessing::name);
}

void DeferredGraphics::setPositionInWindow(const graphicsManager::PositionInWindow& pos) {
    GraphicsInterface::setPositionInWindow(pos);
    linkMember.setPosition(pos);
}

void DeferredGraphics::reset()
{
    if(!CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr")))
    {
        return;
    }

    commandPool = utils::vkDefault::CommandPool(device->device());
    copyCommandBuffers = commandPool.allocateCommandBuffers(resourceCount);

    aDatabase.destroy();
    emptyTextures[Names::blackTexture] = utils::Texture::createEmpty(*device, commandPool, utils::Texture::EmptyType::Black);
    aDatabase.addEmptyTexture(Names::blackTexture, &emptyTextures[Names::blackTexture]);
    emptyTextures[Names::whiteTexture] = utils::Texture::createEmpty(*device, commandPool, utils::Texture::EmptyType::White);
    aDatabase.addEmptyTexture(Names::whiteTexture, &emptyTextures[Names::whiteTexture]);
    if (depthMaps.find(Names::nullDepthMapKey) == depthMaps.end()) {
        const utils::vkDefault::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, {1,1}, VK_SAMPLE_COUNT_1_BIT };
        depthMaps[Names::nullDepthMapKey] = utils::DepthMap(*device, commandPool, shadowsInfo);
        depthMaps[Names::nullDepthMapKey].update();
    }

    createGraphicsPasses();
    createStages();
}

void DeferredGraphics::createGraphicsPasses()
{
    if (!CHECK_M(commandPool, std::string("[ DeferredGraphics ] VkCommandPool is VK_NULL_HANDLE")) ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr")) ||
        !CHECK_M(params.cameraObject, std::string("[ DeferredGraphics ] camera is nullptr"))) 
    {
        return;
    }

    utils::vkDefault::ImageInfo imageInfo{};
	    imageInfo.Count = resourceCount;
        imageInfo.Extent = VkExtent2D{ params.extent[0], params.extent[1] };
	    imageInfo.Samples = params.MSAASamples;

    utils::vkDefault::ImageInfo shadowsInfo{};
        shadowsInfo.Count = resourceCount;
        shadowsInfo.Extent = VkExtent2D{ params.shadowsExtent[0], params.shadowsExtent[1] };
        shadowsInfo.Samples = VK_SAMPLE_COUNT_1_BIT;

    graphicsParams.in.camera = &params.cameraObject;
    graphicsParams.in.nullDepthMapKey = Names::nullDepthMapKey;
    graphicsParams.out.image = Names::MainGraphics::image;
    graphicsParams.out.bloom = Names::MainGraphics::bloom;
    graphicsParams.out.normal = Names::MainGraphics::GBuffer::normal;
    graphicsParams.out.color = Names::MainGraphics::GBuffer::color;
    graphicsParams.out.emission = Names::MainGraphics::GBuffer::emission;
    graphicsParams.out.depth = Names::MainGraphics::GBuffer::depth;
    graphicsParams.shadersPath = params.shadersPath;
    graphicsParams.imageInfo = imageInfo;
    graphicsParams.disableFaceCulling = params.disableFaceCulling;

	const auto l0 = layerPrefix(LayerIndex(0));

    skyboxParams.in.camera = &params.cameraObject;
    skyboxParams.out.baseColor = Names::Skybox::color;
    skyboxParams.out.bloom = Names::Skybox::bloom;
    skyboxParams.shadersPath = params.workflowsShadersPath;
    skyboxParams.imageInfo = imageInfo;

    scatteringParams.in.camera = &params.cameraObject;
    scatteringParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    scatteringParams.out.scattering = Names::Scattering::output;
    scatteringParams.shadersPath = params.workflowsShadersPath;
    scatteringParams.imageInfo = imageInfo;

    SSLRParams.in.camera = &params.cameraObject;
    SSLRParams.in.normal = l0 + Names::MainGraphics::GBuffer::normal;
    SSLRParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    SSLRParams.in.color = l0 + Names::MainGraphics::image;
    SSLRParams.out.sslr = Names::SSLR::output;
    SSLRParams.shadersPath = params.workflowsShadersPath;
    SSLRParams.imageInfo = imageInfo;

    SSAOParams.in.camera = &params.cameraObject;
    SSAOParams.in.normal = l0 + Names::MainGraphics::GBuffer::normal;
    SSAOParams.in.color = l0 + Names::MainGraphics::GBuffer::color;
    SSAOParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    SSAOParams.in.defaultDepthTexture = Names::whiteTexture;
    SSAOParams.out.ssao = Names::SSAO::output;
    SSAOParams.shadersPath = params.workflowsShadersPath;
    SSAOParams.imageInfo = imageInfo;

    layersCombinerParams.in.camera = &params.cameraObject;
    layersCombinerParams.in.color = Names::MainGraphics::image;
    layersCombinerParams.in.bloom = Names::MainGraphics::bloom;
    layersCombinerParams.in.normal = Names::MainGraphics::GBuffer::normal;
    layersCombinerParams.in.depth = Names::MainGraphics::GBuffer::depth;
    layersCombinerParams.in.skyboxColor = Names::Skybox::color;
    layersCombinerParams.in.skyboxBloom = Names::Skybox::bloom;
    layersCombinerParams.in.scattering = Names::Scattering::output;
    layersCombinerParams.in.sslr = Names::SSLR::output;
    layersCombinerParams.in.ssao = Names::SSAO::output;
    layersCombinerParams.in.defaultDepthTexture = Names::whiteTexture;
    layersCombinerParams.in.whiteTexture = Names::whiteTexture;
    layersCombinerParams.out.color = Names::LayersCombiner::color;
    layersCombinerParams.out.bloom = Names::LayersCombiner::bloom;
    layersCombinerParams.out.blur = Names::LayersCombiner::blur;
    layersCombinerParams.layersCount = LayerIndex(params.layersCount);
    layersCombinerParams.shadersPath = params.shadersPath;
    layersCombinerParams.imageInfo = imageInfo;

    bloomParams.in.bloom = Names::LayersCombiner::bloom;
    bloomParams.out.bloom = Names::Bloom::output;
    bloomParams.shadersPath = params.workflowsShadersPath;
    bloomParams.imageInfo = imageInfo;

    blurParams.in.color = Names::LayersCombiner::color;
    blurParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    blurParams.out.blur = Names::Blur::output;
    blurParams.shadersPath = params.workflowsShadersPath;
    blurParams.imageInfo = imageInfo;

    bbParams.in.camera = &params.cameraObject;
    bbParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    bbParams.in.defaultDepthTexture = Names::whiteTexture;
    bbParams.out.boundingBox = Names::BoundingBox::output;
    bbParams.shadersPath = params.workflowsShadersPath;
    bbParams.imageInfo = imageInfo;

    selectorParams.in.normal = l0 + Names::MainGraphics::GBuffer::normal;
    selectorParams.in.depth = l0 + Names::MainGraphics::GBuffer::depth;
    selectorParams.in.defaultDepthTexture = Names::whiteTexture;
    selectorParams.shadersPath = params.workflowsShadersPath;
    selectorParams.imageInfo = imageInfo;

    postProcessingParams.in.baseColor = Names::LayersCombiner::color;
    postProcessingParams.in.bloom = Names::Bloom::output;
    postProcessingParams.in.blur = Names::Blur::output;
    postProcessingParams.in.boundingBox = Names::BoundingBox::output;
    postProcessingParams.out.postProcessing = Names::PostProcessing::output;
    postProcessingParams.shadersPath = params.workflowsShadersPath;
    postProcessingParams.imageInfo = imageInfo;

    shadowGraphicsParameters.imageInfo = shadowsInfo;
    shadowGraphicsParameters.shadersPath = params.workflowsShadersPath,

    workflows.clear();

    for (LayerIndex i{ 0 }; i < LayerIndex(params.layersCount); i++) {
        const auto key = layerPrefix<workflows::WorkflowName>(i) + Names::MainGraphics::name;
        workflows[key] = std::make_unique<Graphics>(graphicsParams, i, &objects, &lights, &depthMaps);
    };
    workflows[Names::LayersCombiner::name] = std::make_unique<LayersCombiner>(layersCombinerParams);
    workflows[Names::PostProcessing::name] = std::make_unique<workflows::PostProcessingGraphics>(postProcessingParams);
    workflows[Names::Blur::name] = std::make_unique<workflows::GaussianBlur>(blurParams);
    workflows[Names::Bloom::name] = std::make_unique<workflows::BloomGraphics>(bloomParams);
    workflows[Names::Skybox::name] = std::make_unique<workflows::SkyboxGraphics>(skyboxParams, &objects);
    workflows[Names::SSLR::name] = std::make_unique<workflows::SSLRGraphics>(SSLRParams);
    workflows[Names::SSAO::name] = std::make_unique<workflows::SSAOGraphics>(SSAOParams);
    workflows[Names::Shadow::name] = std::make_unique<workflows::ShadowGraphics>(shadowGraphicsParameters, &objects, &depthMaps);
    workflows[Names::Scattering::name] = std::make_unique<workflows::Scattering>(scatteringParams, &lights, &depthMaps);
    workflows[Names::BoundingBox::name] = std::make_unique<workflows::BoundingBoxGraphics>(bbParams, &objects);
    workflows[Names::Selector::name] = std::make_unique<workflows::SelectorGraphics>(selectorParams, &params.cursor);

    for(auto& [name, workflow]: workflows){
        workflow->setDeviceProp(*device, device->device());
        if (params.workflowsToAllocate.find(name) != params.workflowsToAllocate.end()) {
            workflow->create(commandPool, aDatabase);
        }
    }

    for (auto& [_, workflow] : workflows) {
        workflow->updateDescriptors(bDatabase, aDatabase);
		workflow->raiseUpdateFlags();
    }

    utils::vkDefault::ImageInfo linkInfo{resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, VK_SAMPLE_COUNT_1_BIT };
    linkMember = DeferredLink(device->device(), params.shadersPath, linkInfo, pRenderPass, position, aDatabase.get(postProcessingParams.out.postProcessing));
}

void DeferredGraphics::createStages()
{
    if (!CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr")))
    {
        return;
    }

    nodes.clear();
    nodes.reserve(6);

    auto getCommandBuffers = [&](const std::vector<workflows::WorkflowName>& names) -> std::vector<const utils::vkDefault::CommandBuffers*> {
        std::vector<const utils::vkDefault::CommandBuffers*> res;
        for (const auto& name : names) {
            if (params.workflowsToAllocate.find(name) == params.workflowsToAllocate.end()) {
                continue;
            }
            if (workflows.find(name) == workflows.end()) {
                continue;
            }
            if (!CHECK_M(workflows[name].get(), std::string("[ DeferredGraphics::createStages ] workflows is nullptr"))) {
                continue;
            }
            const utils::vkDefault::CommandBuffers* commandBuffers = *workflows[name];
            if (!CHECK_M(commandBuffers, std::string("[ DeferredGraphics::createStages ] commandBuffers is nullptr"))) {
                continue;
            }
            res.push_back(commandBuffers);
        }
        return res;
	};

    auto push = [&](utils::PipelineStages& stages, utils::PipelineStage&& stage) {
        if (!stage.empty()) {
			stages.push_back(std::move(stage));
        }
    };

    utils::PipelineStages postProcessingStages;
    push(postProcessingStages, 
        utils::PipelineStage(
            getCommandBuffers({ Names::Selector::name, Names::Bloom::name, Names::Blur::name, Names::BoundingBox::name, Names::PostProcessing::name }),
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            device->device()(0, 0)
        )
    );
    nodes.push_back(utils::PipelineNode(device->device(), std::move(postProcessingStages), nullptr));

    utils::PipelineStages layersCombinerStages;
    push(layersCombinerStages, utils::PipelineStage(getCommandBuffers({ Names::LayersCombiner::name }), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0, 0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(layersCombinerStages), &nodes.back()));

    utils::PipelineStages preCombinedStages;
    push(preCombinedStages, utils::PipelineStage(getCommandBuffers({ Names::Scattering::name }), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0, 0)));
    push(preCombinedStages, utils::PipelineStage(getCommandBuffers({ Names::SSLR::name }), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    push(preCombinedStages, utils::PipelineStage(getCommandBuffers({ Names::SSAO::name }), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0, 0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(preCombinedStages), &nodes.back()));

    std::vector<workflows::WorkflowName> deferredStagesNames;
    for (LayerIndex i{ 0 }; i < LayerIndex(params.layersCount); i++) {
        const auto key = layerPrefix<workflows::WorkflowName>(i) + Names::MainGraphics::name;
        deferredStagesNames.push_back(key);
    };
    utils::PipelineStages deferredStages;
    push(deferredStages, utils::PipelineStage(getCommandBuffers(deferredStagesNames), VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(deferredStages), &nodes.back()));

    utils::PipelineStages prepareStages;
    push(prepareStages, utils::PipelineStage(getCommandBuffers({Names::Shadow::name}), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->device()(0,0)));
    push(prepareStages, utils::PipelineStage(getCommandBuffers({Names::Skybox::name}), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(prepareStages), &nodes.back()));

    utils::PipelineStages copyStages;
    push(copyStages, utils::PipelineStage({ &copyCommandBuffers }, VK_PIPELINE_STAGE_TRANSFER_BIT, device->device()(0,0)));
    nodes.push_back(utils::PipelineNode(device->device(), std::move(copyStages), &nodes.back()));
}

utils::vkDefault::VkSemaphores DeferredGraphics::submit(utils::ResourceIndex resourceIndex, const utils::vkDefault::VkSemaphores& externalSemaphore) {
    return nodes.back().submit(resourceIndex, externalSemaphore);
}

void DeferredGraphics::updateParameters() {
    if (params.blurDepth().isDirty()) {
        const auto blurDepth = params.blurDepth().consume();
        blurParams.blurDepth = blurDepth;
        requestUpdate(Names::Blur::name);
    }
    if (params.bloomThreshold().isDirty()) {
        const auto bloomThreshold = params.bloomThreshold().consume();
        layersCombinerParams.bloomThreshold = bloomThreshold;
        requestUpdate(Names::LayersCombiner::name);
    }
    if (params.minAmbientFactor().isDirty()) {
        const auto ao = params.minAmbientFactor().consume();
        graphicsParams.minAmbientFactor = ao;
        requestUpdate(Names::MainGraphics::name);
    }
}

void DeferredGraphics::update(utils::ResourceIndex resourceIndex) {
    updateParameters();

    const auto index = resourceIndex.get();
    CHECK(copyCommandBuffers[index].reset());
    CHECK(copyCommandBuffers[index].begin());
    if (params.cameraObject) {
        params.cameraObject->setViewport(float(params.extent[0]), float(params.extent[1]));
        params.cameraObject->update(resourceIndex, copyCommandBuffers[index]);
    }
    for (auto& object : objects) {
        if (!object) continue;
        object->update(resourceIndex, copyCommandBuffers[index]);
    }
    for (auto& light : lights) {
        if (!light) continue;
        light->update(resourceIndex, copyCommandBuffers[index]);
    }
    CHECK(copyCommandBuffers[index].end());

    for (auto& [name, workflow] : workflows) {
        if (!workflow) continue;
        workflow->update(resourceIndex);
    }
}

void DeferredGraphics::create(interfaces::Model *pModel){
    if (!CHECK_M(pModel, "[ DeferredGraphics::create ] : pModel is nullptr") || 
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return;
    }

    pModel->create(*device, commandPool);
}

void DeferredGraphics::bind(interfaces::Camera* cameraObject){
    if (!CHECK_M(cameraObject, "[ DeferredGraphics::bind ] : cameraObject is nullptr") ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return;
    }

    params.cameraObject = cameraObject;
    cameraObject->create(*device, resourceCount);
}

bool DeferredGraphics::remove(interfaces::Camera* cameraObject){
    if (!CHECK_M(cameraObject, "[ DeferredGraphics::remove ] : cameraObject is nullptr") ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return false;
    }

    bool res = params.cameraObject == cameraObject;
    if(res){
        params.cameraObject = nullptr;
    }
    return res;
}

void DeferredGraphics::bind(interfaces::Light* lightSource){
    if (!CHECK_M(lightSource, "[ DeferredGraphics::bind ] : bind is nullptr") ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return;
    }

    lightSource->create(*device, commandPool, resourceCount);
    lights.push_back(lightSource);

    if (depthMaps.find(lightSource) == depthMaps.end()) {
        const auto lightProps = lightSource->lightMask().property();
        if (lightProps.has(interfaces::LightProperty::enableShadow)) {
            const VkExtent2D shadowsExtent{ params.shadowsExtent[0], params.shadowsExtent[1] };
            const utils::vkDefault::ImageInfo shadowsInfo{ resourceCount, VK_FORMAT_D32_SFLOAT, shadowsExtent, VK_SAMPLE_COUNT_1_BIT };
            depthMaps[lightSource] = utils::DepthMap(*device, commandPool, shadowsInfo);
            depthMaps[lightSource].update();
        }
    }

    requestUpdate(Names::Shadow::name);
    requestUpdate(Names::Scattering::name);
    requestUpdate(Names::MainGraphics::name);
}

bool DeferredGraphics::remove(interfaces::Light* lightSource){
    if (!CHECK_M(lightSource, "[ DeferredGraphics::remove ] : lightSource is nullptr")) {
        return false;
    }
    
    size_t size = lights.size();
    lights.erase(std::remove(lights.begin(), lights.end(), lightSource), lights.end());

    if(depthMaps.find(lightSource) != depthMaps.end()){
        depthMaps.erase(lightSource);
        requestUpdate(Names::Shadow::name);
    }

    requestUpdate(Names::Scattering::name);
    requestUpdate(Names::MainGraphics::name);
    return size - lights.size() > 0;
}

void DeferredGraphics::bind(interfaces::Object* object){
    if (!CHECK_M(object, "[ DeferredGraphics::bind ] : object is nullptr") ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return;
    }

    object->create(*device, commandPool, resourceCount);
    objects.push_back(object);

    requestUpdate(Names::Skybox::name);
    requestUpdate(Names::Shadow::name);
    requestUpdate(Names::BoundingBox::name);
    requestUpdate(Names::MainGraphics::name);
}

bool DeferredGraphics::remove(interfaces::Object* object){
    if (!CHECK_M(object, "[ DeferredGraphics::remove ] : object is nullptr")) {
        return false;
    }

    size_t size = objects.size();
    objects.erase(std::remove(objects.begin(), objects.end(), object), objects.end());

    requestUpdate(Names::Skybox::name);
    requestUpdate(Names::Shadow::name);
    requestUpdate(Names::BoundingBox::name);
    requestUpdate(Names::MainGraphics::name);
    return size - objects.size() > 0;
}

void DeferredGraphics::bind(utils::Cursor* cursor) {
    if (!CHECK_M(cursor, "[ DeferredGraphics::bind ] : cursor is nullptr") ||
        !CHECK_M(device, std::string("[ DeferredGraphics ] device is nullptr"))) {
        return;
    }
    params.cursor = cursor;
    cursor->create(*device, device->device());
    requestUpdate(Names::Selector::name);
}

bool DeferredGraphics::remove(utils::Cursor* cursor) {
    if (!CHECK_M(cursor, "[ DeferredGraphics::remove ] : cursor is nullptr")) {
        return false;
    }

	bool res = params.cursor == cursor;
    if (res) {
        params.cursor = nullptr;
    }
	return res;
}

DeferredGraphics& DeferredGraphics::requestUpdate(const workflows::WorkflowName& name) {
    auto raiseUpdateFlags = [&](const workflows::WorkflowName& workflowName) {
        if (workflows.find(workflowName) == workflows.end()) {
            return;
        }
        auto workflow = workflows[workflowName].get();
        if (CHECK_M(workflow, "[ DeferredGraphics::requestUpdate ] : " + std::string(workflowName) + " is nullptr")) {
            workflow->raiseUpdateFlags();
        }
    };

    if (name == Names::MainGraphics::name) {
        for (LayerIndex i{ 0 }; i < LayerIndex(params.layersCount); i++) {
            const auto key = layerPrefix<workflows::WorkflowName>(i) + Names::MainGraphics::name;
            raiseUpdateFlags(key);
        };
    }
    else {
        raiseUpdateFlags(name);
    }
    return *this;
}

DeferredGraphics& DeferredGraphics::requestUpdate() {
    for (auto& [_, workflow] : workflows) {
        workflow->raiseUpdateFlags();
    }
    return *this;
}

DeferredGraphics& DeferredGraphics::setEnable(const workflows::ParameterName& name, bool enable){
    if (workflowsParameters.find(name) == workflowsParameters.end()) {
        return *this;
    }
    if (auto& workflowsParameter = workflowsParameters[name]; workflowsParameter) {
        workflowsParameter->enable = enable;
    }
    return *this;
}

bool DeferredGraphics::getEnable(const workflows::ParameterName& name){
    if (workflowsParameters.find(name) == workflowsParameters.end()) {
        return false;
    }
    if (auto& workflowsParameter = workflowsParameters[name]; workflowsParameter) {
        return workflowsParameter->enable;
    }
    return false;
}

Parameters& DeferredGraphics::parameters() {
    return params;
}

workflows::BloomParameters& DeferredGraphics::bloomWorkflowParams() {
    return bloomParams;
}

workflows::ScatteringParameters& DeferredGraphics::scatteringWorkflowParams() {
    return scatteringParams;
}

workflows::SSAOParameters& DeferredGraphics::ssaoWorkflowParams() {
    return SSAOParams;
}

void DeferredGraphics::draw(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) const {
    linkMember.draw(commandBuffer, resourceIndex);
}

} // moon::deferredGraphics
