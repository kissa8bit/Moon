#include "testScene.h"

#include <deferredGraphics/deferredGraphics.h>
#include <graphicsManager/graphicsManager.h>

#include <math/linearAlgebra.h>

#include <models/gltfmodel.h>
#include <models/plymodel.h>

#include <transformationals/cameras.h>
#include <transformationals/objects.h>
#include <transformationals/group.h>

#include <utils/cursor.h>

#include <interfaces/light.h>

#include <entities/baseCamera.h>
#include <entities/spotLight.h>
#include <entities/baseObject.h>
#include <entities/skyboxObject.h>

#ifdef IMGUI_GRAPHICS
#include "testSceneGui.h"
#include "imgui_impl_glfw.h"
#endif

#include <random>
#include <limits>
#include <cstring>
#include <algorithm>
#include <execution>

testScene::testScene(moon::graphicsManager::GraphicsManager& app, moon::tests::GlfwWindow& window, const std::filesystem::path& ExternalPath):
    ExternalPath(ExternalPath),
    window(window),
    app(app),
    mouse(window),
    board(std::make_shared<controller>(window, glfwGetKey))
{
    mouse.control->sensitivity = 0.01f;
    board->sensitivity = 0.05f;
    create();
}

testScene::~testScene() {
#ifdef IMGUI_GRAPHICS
    ImGui_ImplGlfw_Shutdown();
#endif
}

void testScene::resize() {
    if(auto pBaseCamera = dynamic_cast<moon::entities::BaseCamera*>(cameras["base"].get()); pBaseCamera){
        pBaseCamera->setProjMatrix(moon::math::perspective(moon::math::radians(45.0f), window.aspectRatio(), 0.1f));
    }
    graphics["base"]->parameters().extent = window.sizes();

#ifdef SECOND_VIEW_WINDOW
    cameras["view"]->setProjMatrix(moon::math::perspective(moon::math::radians(45.0f), window.aspectRatio(), 0.1f));
    graphics["view"]->parameters().extent = window.sizes() / 3;
#endif

    for(auto& [_,graph]: graphics){
        graph->reset();
    }
}

void testScene::create()
{
    cameras["base"] = std::make_shared<moon::entities::BaseCamera>(45.0f, window.aspectRatio(), 0.1f);
    cameras["base"]->translate({0.0f, 0.0f, 10.0f});
    moon::deferredGraphics::Parameters deferredGraphicsParameters;
    deferredGraphicsParameters.shadersPath = ExternalPath / "core/deferredGraphics/spv";
    deferredGraphicsParameters.workflowsShadersPath = ExternalPath / "core/workflows/spv";
    deferredGraphicsParameters.extent = window.sizes();
	deferredGraphicsParameters.layersCount() = 3;
	deferredGraphicsParameters.minAmbientFactor() = 0.01f;
    graphics["base"] = std::make_shared<moon::deferredGraphics::DeferredGraphics>(deferredGraphicsParameters);
    app.setGraphics(graphics["base"].get());
    graphics["base"]->bind(cameras["base"]->camera());
    graphics["base"]->bind(mouse);
    graphics["base"]->
        setEnable(moon::deferredGraphics::Names::Skybox::param, true).
        setEnable(moon::deferredGraphics::Names::Blur::param, true).
        setEnable(moon::deferredGraphics::Names::Bloom::param, true).
        setEnable(moon::deferredGraphics::Names::SSAO::param, false).
        setEnable(moon::deferredGraphics::Names::SSLR::param, false).
        setEnable(moon::deferredGraphics::Names::Scattering::param, true).
        setEnable(moon::deferredGraphics::Names::Shadow::param, true).
        setEnable(moon::deferredGraphics::Names::Selector::param, true);
    graphics["base"]->reset();

#ifdef SECOND_VIEW_WINDOW
    cameras["view"] = std::make_shared<moon::transformational::Camera>(45.0f, window.aspectRatio(), 0.1f);
    deferredGraphicsParameters.extent /= 3;
    graphics["view"] = std::make_shared<moon::deferredGraphics::DeferredGraphics>(deferredGraphicsParameters);
    graphics["view"]->setPositionInWindow({ { 0.5f, 0.5f }, { 0.33f, 0.33f } });
    app.setGraphics(graphics["view"].get());
    graphics["view"]->bind(*cameras["view"].get());
    graphics["view"]->
        setEnable("TransparentLayer", true).
        setEnable("Skybox", false).
        setEnable("Blur", false).
        setEnable("Bloom", false).
        setEnable("SSAO", false).
        setEnable("SSLR", false).
        setEnable("Scattering", false).
        setEnable("Shadow", false);
    graphics["view"]->reset();
#endif

#ifdef IMGUI_GRAPHICS
    gui = std::make_shared<moon::imguiGraphics::ImguiGraphics>(app.getInstance(), app.getImageCount());
    app.setGraphics(gui.get());
    if(ImGui_ImplGlfw_InitForVulkan(window, true)) {
        gui->reset();
    }
#endif

    createModels();
    createObjects();
    createLight();

    groups["lightBox"]->translate({0.0f,0.0f,25.0f});
    groups["ufo0"]->translate({5.0f,0.0f,5.0f});
    groups["ufo1"]->translate({-5.0f,0.0f,5.0f});
    groups["ufo2"]->translate({10.0f,0.0f,5.0f});
    groups["ufo3"]->translate({-10.0f,0.0f,5.0f});
}

void testScene::requestUpdate() {
    if (graphics["base"]) {
        graphics["base"]->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
    }
#ifdef SECOND_VIEW_WINDOW
    if (graphics["view"]) {
        graphics["view"]->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
    }
#endif
}

void testScene::makeGui() {
    if (ImGui::TreeNodeEx("General", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Update")) {
            window.windowResized() = true;
        }

        ImGui::SetNextItemWidth(150.0f);
        ImGui::SliderFloat("mouse sensitivity", &mouse.control->sensitivity, 0.0f, 5.0f);

        ImGui::SetNextItemWidth(150.0f);
        ImGui::SliderFloat("board sensitivity", &board->sensitivity, 0.0f, 5.0f);

        ImGui::TreePop();
    }

    if(ImGui::TreeNodeEx("Screenshot", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
        moon::tests::gui::makeScreenshot("Make screenshot", app);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Performance", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
        moon::tests::gui::fpsPlot(1.0f / frameTime);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Base Graphics Settings", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::BeginGroup();
            ImGui::Text("props : "); ImGui::SameLine(); ImGui::Separator();
            moon::tests::gui::graphicsProps(graphics["base"], mouse.cursor);
        ImGui::EndGroup();

        ImGui::BeginGroup();
            ImGui::Text("pose in window : "); ImGui::SameLine(); ImGui::Separator();
            moon::tests::gui::setPoseInWindow(graphics["base"]);
        ImGui::EndGroup();

        ImGui::BeginGroup();
            ImGui::Text("pipelines : "); ImGui::SameLine(); ImGui::Separator();
            window.windowResized() |= moon::tests::gui::switchers(graphics["base"]);
        ImGui::EndGroup();

        ImGui::TreePop();
    }

#ifdef SECOND_VIEW_WINDOW
    if (ImGui::TreeNodeEx("Second Graphics Settings", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("pose in window : "); ImGui::SameLine(); ImGui::Separator();
        moon::tests::gui::setPoseInWindow(graphics["view"]);

        ImGui::Text("pipelines : "); ImGui::SameLine(); ImGui::Separator();
        window.windowResized() |= moon::tests::gui::switchers(graphics["view"]);

        ImGui::TreePop();
    }
#endif

    if (ImGui::TreeNodeEx("Animation", ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SetNextItemWidth(150.0f);
        ImGui::SliderFloat("animation speed", &animationSpeed, 0.0f, 5.0f);

        ImGui::SetNextItemWidth(150.0f);
        static float changeTime = 0.5f;
        ImGui::SliderFloat("change animation time", &changeTime, 0.0f, 1.0f);

        auto pBaseObject = dynamic_cast<moon::entities::BaseObject*>(controledObject.ptr);

        if (pBaseObject && pBaseObject->animationControl.size() > 0) {
            static const char* animationsList[] = { "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
            int selected = pBaseObject->animationControl.current() + 1;
            ImGui::SetNextItemWidth(150.0f);
            if (ImGui::Combo("animations list", &selected, animationsList, std::min(IM_ARRAYSIZE(animationsList), (int)pBaseObject->animationControl.size() + 1))) {
                pBaseObject->animationControl.set(selected - 1, changeTime);
            }
        }

        ImGui::TreePop();
    }

    if (controledObject)
    {
        std::string groupKey;
        moon::transformational::Group* controlGroup = nullptr;
        for (auto& [key, g] : groups) {
            if (g->find(controledObject.ptr)) {
                controlGroup = g.get();
                groupKey = key;
                break;
            }
        }

        if (ImGui::TreeNodeEx(std::string("Object : " + controledObject.name).c_str(), ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginGroup();
            ImGui::Text("translation : "); ImGui::SameLine(); ImGui::Separator();
            if (controlGroup) {
                moon::tests::gui::transManipulator<0>(*controlGroup, "x_tr");
                moon::tests::gui::transManipulator<1>(*controlGroup, "y_tr");
                moon::tests::gui::transManipulator<2>(*controlGroup, "z_tr");
            }
            else {
                moon::tests::gui::transManipulator<0>(*controledObject.ptr, "x_tr");
                moon::tests::gui::transManipulator<1>(*controledObject.ptr, "y_tr");
                moon::tests::gui::transManipulator<2>(*controledObject.ptr, "z_tr");
            }
            ImGui::EndGroup();

            ImGui::BeginGroup();
            ImGui::Text("scale : "); ImGui::SameLine(); ImGui::Separator();
            if (ImGui::Button("Align by max")) {
                controledObject->scale({ controledObject->scaling().maxValue() });
            }
            ImGui::SameLine(0.0, 10.0);
            if (ImGui::Button("Align by min")) {
                controledObject->scale({ controledObject->scaling().minValue() });
            }
            moon::tests::gui::scaleManipulator<0>(controledObject, "x_sc");
            moon::tests::gui::scaleManipulator<1>(controledObject, "y_sc");
            moon::tests::gui::scaleManipulator<2>(controledObject, "z_sc");
            ImGui::EndGroup();

            ImGui::BeginGroup();
            ImGui::Text("rotation : "); ImGui::SameLine(); ImGui::Separator();
            if (controlGroup) {
                moon::tests::gui::rotationmManipulator(*controlGroup, dynamic_cast<moon::entities::BaseCamera*>(cameras["base"].get()));
                ImGui::SameLine(0.0, 10.0);
                moon::tests::gui::printQuaternion(controlGroup->rotation());
            }
            else {
                moon::tests::gui::rotationmManipulator(*controledObject.ptr, dynamic_cast<moon::entities::BaseCamera*>(cameras["base"].get()));
                ImGui::SameLine(0.0, 10.0);
                moon::tests::gui::printQuaternion(controledObject->rotation());
            }
            ImGui::EndGroup();

            ImGui::BeginGroup();
            ImGui::Text("colors : "); ImGui::SameLine(); ImGui::Separator();
            if (moon::tests::gui::setOutlighting(controledObject)) {
                requestUpdate();
            }
            moon::tests::gui::setColors(controledObject);
            ImGui::EndGroup();

            if (auto model = dynamic_cast<moon::models::PlyModel*>(controledObject.ptr->object()->model()); model) {
                ImGui::BeginGroup();
                ImGui::Text("materials : "); ImGui::SameLine(); ImGui::Separator();
                if (moon::tests::gui::setPlyMaterial(model)) {
                    requestUpdate();
                }
                ImGui::EndGroup();
            }

            ImGui::TreePop();
        }

        if (!groupKey.empty() && ImGui::TreeNodeEx(std::string("Lights : " + controledObject.name).c_str(), ImGuiTreeNodeFlags_::ImGuiTreeNodeFlags_DefaultOpen)) {
            auto requestLightUpdate = [this]() {
                auto& g = *graphics["base"];
                g.requestUpdate(moon::deferredGraphics::Names::Shadow::name);
                g.requestUpdate(moon::deferredGraphics::Names::Scattering::name);
                g.requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
            };

            auto it = groupSpotLights.find(groupKey);
            if (it != groupSpotLights.end()) {
                ImGui::BeginGroup();
                    ImGui::Text("lights : "); ImGui::SameLine(); ImGui::Separator();
                    for (int i = 0; i < static_cast<int>(it->second.size()); ++i) {
                        if (moon::tests::gui::spotLightSliders(*it->second[i], i)) requestLightUpdate();
                        moon::tests::gui::spotLightProjectionSliders(*it->second[i], i);
                    }
                ImGui::EndGroup();
            }

            auto lit = lightPoints.find(groupKey);
            if (lit != lightPoints.end()) {
                ImGui::BeginGroup();
                    ImGui::Text("isotropic light : "); ImGui::SameLine(); ImGui::Separator();
                    if (moon::tests::gui::spotLightSliders(*lit->second, 0)) requestLightUpdate();
                ImGui::EndGroup();
            }
        }

        ImGui::TreePop();
    }
}

void testScene::updateFrame(uint32_t frameNumber, float inFrameTime)
{
    frameTime = inFrameTime;
    glfwPollEvents();

#ifdef IMGUI_GRAPHICS
    ImGuiIO io = ImGui::GetIO();

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetWindowSize({350, 100}, ImGuiCond_::ImGuiCond_Once);

    if (ImGui::Begin("Debug")) {
        makeGui();
    }
    ImGui::End();

    if (!io.WantCaptureMouse)    mouseEvent();
    if (!io.WantCaptureKeyboard) keyboardEvent();
#else
    mouseEvent();
    keyboardEvent();
#endif

    float animationTime = animationSpeed * frameTime;
    static float globalTime = 0.0f;
    globalTime += animationTime;

    skyboxObjects["stars"]->rotate(0.1f * animationTime, normalized(moon::math::vec3(1.0f, 1.0f, 1.0f)));
    objects["helmet"]->rotate(0.5f * animationTime, normalized(moon::math::vec3(0.0f, 0.0f, 1.0f)));
    objects["helmet"]->translation() = moon::math::quat(0.0f, {27.0f, -10.0f, 14.0f + 0.2f * std::sin(globalTime)});
    objects["helmet"]->update();

    std::for_each(std::execution::par_unseq, objects.begin(), objects.end(), [frameNumber, animationTime](auto& object) {
        auto pBaseObject = dynamic_cast<moon::entities::BaseObject*>(object.second.get());
        if(!pBaseObject) return;
        pBaseObject->animationControl.update(frameNumber, animationTime);
    });
}

void testScene::createModels()
{
    const auto resourceCount = app.getResourceCount();
    const std::filesystem::path modelPath = ExternalPath / "dependences/model";
    const auto glTFSamples = modelPath / "glTF-Sample-Models/2.0";

#define MAKE_GLTF std::make_shared<moon::models::GltfModel>
#define MAKE_PLY std::make_shared<moon::models::PlyModel>

    models["bee"] = MAKE_GLTF(modelPath / "glb/Bee.glb", 2 * resourceCount);
    models["butterfly"] = MAKE_GLTF(modelPath / "glb/butterfly.glb", resourceCount);
    models["ufo"] = MAKE_GLTF(modelPath / "glb/RetroUFO.glb");
    models["robot"] = MAKE_GLTF(modelPath / "glb/Robot.glb", resourceCount);
    models["skeleton"] = MAKE_GLTF(modelPath / "glb/skeleton.glb", resourceCount);
    models["octopus"] = MAKE_GLTF(modelPath / "glb/octopus.glb", resourceCount);
    models["SimpleSkin"] = MAKE_GLTF(glTFSamples / "SimpleSkin/glTF/SimpleSkin.gltf", resourceCount);
    models["RiggedFigure"] = MAKE_GLTF(glTFSamples / "RiggedFigure/glTF-Binary/RiggedFigure.glb", resourceCount);
    models["InterpolationTest"] = MAKE_GLTF(glTFSamples / "InterpolationTest/glTF-Binary/InterpolationTest.glb", resourceCount);
    models["RecursiveSkeletons"] = MAKE_GLTF(glTFSamples / "RecursiveSkeletons/glTF-Binary/RecursiveSkeletons.glb", resourceCount);

    models["box"] = MAKE_GLTF(glTFSamples / "Box/glTF-Binary/Box.glb");
    models["sponza"] = MAKE_GLTF(glTFSamples / "Sponza/glTF/Sponza.gltf");
    // models["sponza2"] = MAKE_GLTF(modelPath / "main_sponza/NewSponza_Main_glTF_003.gltf");
    // models["sponza3"] = MAKE_GLTF(modelPath / "pkg_a_curtains/NewSponza_Curtains_glTF.gltf");
    models["duck"] = MAKE_GLTF(glTFSamples / "Duck/glTF-Binary/Duck.glb");
    models["DragonAttenuation"] = MAKE_GLTF(glTFSamples / "DragonAttenuation/glTF-Binary/DragonAttenuation.glb");
    models["DamagedHelmet"] = MAKE_GLTF(glTFSamples / "DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    models["AlphaBlendModeTest"] = MAKE_GLTF(glTFSamples / "AlphaBlendModeTest/glTF-Binary/AlphaBlendModeTest.glb");

    models["teapot"] = MAKE_PLY(modelPath / "ply/teapot.ply");

    for(auto& [_,model]: models){
        graphics["base"]->create(model.get());
    }
}

void testScene::createObjects()
{
    auto resourceCount = app.getResourceCount();
    staticObjects["sponza"] = std::make_shared<moon::entities::BaseObject>(models["sponza"].get());
    staticObjects["sponza"]->rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f}).scale({3.0f,3.0f,3.0f});

    // staticObjects["sponza2"] = std::make_shared<moon::entities::BaseObject>(models["sponza2"].get());
    // staticObjects["sponza2"]->rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f });
    // 
    // staticObjects["sponza3"] = std::make_shared<moon::entities::BaseObject>(models["sponza3"].get());
    // staticObjects["sponza3"]->rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f });

    objects["bee0"] = std::make_shared<moon::entities::BaseObject>(models["bee"].get(), 0, resourceCount);
    objects["bee0"]->translate({5.0f,0.0f,0.0f}).rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f});
    static_cast<moon::entities::BaseObject*>(objects["bee0"].get())->animationControl.set(0);

    objects["bee1"] = std::make_shared<moon::entities::BaseObject>(models["bee"].get(), resourceCount, resourceCount);
    objects["bee1"]->translate({-5.0f,0.0f,0.0f}).rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f}).scale({0.2f,0.2f,0.2f});
    static_cast<moon::entities::BaseObject*>(objects["bee1"].get())->setColor(moon::math::vec4(0.0f, 0.0f, 0.0f, -0.7f)).animationControl.set(1);

    objects["butterfly"] = std::make_shared<moon::entities::BaseObject>(models["butterfly"].get(), 0, resourceCount);
    objects["butterfly"]->rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).scale({ 20.2f,20.2f,20.2f });
    static_cast<moon::entities::BaseObject*>(objects["butterfly"].get())->animationControl.set(0);

    objects["duck"] = std::make_shared<moon::entities::BaseObject>(models["duck"].get());
    objects["duck"]->translate({0.0f,6.3f,12.1f}).rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f}).rotate(moon::math::radians(-45.0f), { 0.0f,0.0f,1.0f });
    static_cast<moon::entities::BaseObject*>(objects["duck"].get())->setColor(moon::math::vec4(0.0f,0.0f,0.0f,-0.8f));

    objects["lightBox"] = std::make_shared<moon::entities::BaseObject>(models["box"].get());
    static_cast<moon::entities::BaseObject*>(objects["lightBox"].get())->setBloom(moon::math::vec4(1.0f,1.0f,1.0f,1.0f));
    groups["lightBox"] = std::make_shared<moon::transformational::Group>();
    groups["lightBox"]->add(objects["lightBox"].get());

    objects["dragon"] = std::make_shared<moon::entities::BaseObject>(models["DragonAttenuation"].get());
    objects["dragon"]->scale(1.0f).rotate(moon::math::quat(0.5f, 0.5f, -0.5f, -0.5f)).translate(moon::math::vec3(26.0f, 11.0f, 11.0f));

    objects["helmet"] = std::make_shared<moon::entities::BaseObject>(models["DamagedHelmet"].get());
    objects["helmet"]->scale(1.0f).rotate(moon::math::quat(0.5f, 0.5f, -0.5f, -0.5f));

    objects["octopus"] = std::make_shared<moon::entities::BaseObject>(models["octopus"].get(), 0, resourceCount);
    objects["octopus"]->rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).scale({ 3.0f,3.0f,3.0f }).translate({27.0f, -2.0f, 10.0f});
    static_cast<moon::entities::BaseObject*>(objects["octopus"].get())->animationControl.set(0);

    objects["robot"] = std::make_shared<moon::entities::BaseObject>(models["robot"].get(), 0, resourceCount);
    objects["robot"]->scale(25.0f).rotate(moon::math::quat(0.5f, 0.5f, -0.5f, -0.5f)).rotate(moon::math::radians(180.0f), {0.0f, 0.0f, 1.0f}).translate(moon::math::vec3(-30.0f, 11.0f, 10.0f));
    static_cast<moon::entities::BaseObject*>(objects["robot"].get())->animationControl.set(1);

    objects["skeleton"] = std::make_shared<moon::entities::BaseObject>(models["skeleton"].get(), 0, resourceCount);
    objects["skeleton"]->scale(3.0f).rotate(moon::math::quat(0.5f, 0.5f, -0.5f, -0.5f)).rotate(moon::math::radians(180.0f), { 0.0f, 0.0f, 1.0f }).translate(moon::math::vec3(-30.0f, 6.0f, 10.0f));
    static_cast<moon::entities::BaseObject*>(objects["skeleton"].get())->animationControl.set(0);

    objects["RecursiveSkeletons"] = std::make_shared<moon::entities::BaseObject>(models["RecursiveSkeletons"].get());
    objects["RecursiveSkeletons"]->scale(0.02f).rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).translate(moon::math::vec3(-30.0f, -10.0f, 10.0f));
    static_cast<moon::entities::BaseObject*>(objects["RecursiveSkeletons"].get())->animationControl.set(0);

    objects["SimpleSkin"] = std::make_shared<moon::entities::BaseObject>(models["SimpleSkin"].get());
    objects["SimpleSkin"]->scale(2.0f).rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).rotate(moon::math::radians(90.0f), { 0.0f,0.0f,1.0f }).translate(moon::math::vec3(-34.0f, -7.0f, 10.0f));
    static_cast<moon::entities::BaseObject*>(objects["SimpleSkin"].get())->animationControl.set(0);

    objects["RiggedFigure"] = std::make_shared<moon::entities::BaseObject>(models["RiggedFigure"].get());
    objects["RiggedFigure"]->scale(1.0f).rotate(moon::math::radians(-90.0f), { 1.0f,0.0f,0.0f }).rotate(moon::math::radians(90.0f), { 0.0f,0.0f,1.0f }).translate(moon::math::vec3(-32.0f, -5.0f, 10.0f));
    static_cast<moon::entities::BaseObject*>(objects["RiggedFigure"].get())->animationControl.set(0);

    objects["InterpolationTest"] = std::make_shared<moon::entities::BaseObject>(models["InterpolationTest"].get());
    objects["InterpolationTest"]->scale(0.5f).rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).rotate(moon::math::radians(90.0f), { 0.0f,0.0f,1.0f }).translate(moon::math::vec3(-32.0f, 0.0f, 13.0f));
    static_cast<moon::entities::BaseObject*>(objects["InterpolationTest"].get())->animationControl.set(0);

    objects["AlphaBlendModeTest"] = std::make_shared<moon::entities::BaseObject>(models["AlphaBlendModeTest"].get());
    objects["AlphaBlendModeTest"]->rotate(moon::math::radians(90.0f), { 1.0f,0.0f,0.0f }).rotate(moon::math::radians(-90.0f), { 0.0f,0.0f,1.0f }).translate({-24.0f, 3.25f, 12.2f}).scale({0.3f,0.3f, 0.3f});

    objects["teapot"] = std::make_shared<moon::entities::BaseObject>(models["teapot"].get());
    objects["teapot"]->translate({ 21.0f, 3.25f, 12.2f }).scale({ 0.5f });
    static_cast<moon::entities::BaseObject*>(objects["teapot"].get())->setColor(moon::math::vec4(0.5f, 0.5f, 0.5f, 1.0f));

    auto makeUfoLightObject = [&](const std::string& key, auto setupGroup) {
        objects[key] = std::make_shared<moon::entities::BaseObject>(models["ufo"].get());
        objects[key]->rotate(moon::math::radians(90.0f), {1.0f, 0.0f, 0.0f});
        groups[key] = std::make_shared<moon::transformational::Group>();
        setupGroup(*groups[key]);
        groups[key]->add(objects[key].get());
    };

    makeUfoLightObject("ufo_light_0", [](auto& g) {
        g.rotate(moon::math::radians(45.0f), moon::math::vec3( 1.0f, 0.0f,  0.0f))
         .rotate(moon::math::radians(45.0f), moon::math::vec3( 0.0f, 0.0f, -1.0f))
         .translate(moon::math::vec3(24.0f, 7.5f, 18.0f));
    });
    makeUfoLightObject("ufo_light_1", [](auto& g) {
        g.rotate(moon::math::radians(45.0f), moon::math::vec3(-1.0f, 0.0f, 0.0f))
         .rotate(moon::math::radians(45.0f), moon::math::vec3( 0.0f, 0.0f, 1.0f))
         .translate(moon::math::vec3(24.0f, -7.5f, 18.0f));
    });
    makeUfoLightObject("ufo_light_2", [](auto& g) {
        g.rotate(moon::math::radians(30.0f), moon::math::vec3(-1.0f, 0.0f, 0.0f))
         .rotate(moon::math::radians(30.0f), moon::math::vec3( 0.0f, 0.0f, 1.0f))
         .translate(moon::math::vec3(-32.0f, 13.0f, 19.0f));
    });
    makeUfoLightObject("ufo_light_3", [](auto& g) {
        g.rotate(moon::math::radians(30.0f), moon::math::vec3( 1.0f, 0.0f,  0.0f))
         .rotate(moon::math::radians(30.0f), moon::math::vec3( 0.0f, 0.0f, -1.0f))
         .translate(moon::math::vec3(-32.0f, 7.0f, 19.0f));
    });
    makeUfoLightObject("ufo_light_4", [](auto& g) {
        g.rotate(moon::math::radians(30.0f), moon::math::vec3(-1.0f, 0.0f,  0.0f))
         .rotate(moon::math::radians(30.0f), moon::math::vec3( 0.0f, 0.0f, -1.0f))
         .translate(moon::math::vec3(-26.0f, 13.0f, 19.0f));
    });

    for (int i = 0; i < 4; i++) {
        const auto key = "ufo" + std::to_string(i);
        objects[key] = std::make_shared<moon::entities::BaseObject>(models["ufo"].get());
        objects[key]->rotate(moon::math::radians(90.0f), {1.0f, 0.0f, 0.0f});
        static_cast<moon::entities::BaseObject*>(objects[key].get())->setColor(moon::math::vec4(0.0f, 0.0f, 0.0f, -0.8f));
        groups[key] = std::make_shared<moon::transformational::Group>();
        groups[key]->add(objects[key].get());
    }

    skyboxObjects["lake"] = std::make_shared<moon::entities::SkyboxObject>(
        moon::utils::vkDefault::Paths{
            ExternalPath / "dependences/texture/skybox/left.jpg",
            ExternalPath / "dependences/texture/skybox/right.jpg",
            ExternalPath / "dependences/texture/skybox/front.jpg",
            ExternalPath / "dependences/texture/skybox/back.jpg",
            ExternalPath / "dependences/texture/skybox/top.jpg",
            ExternalPath / "dependences/texture/skybox/bottom.jpg"
    });
    skyboxObjects["lake"]->scale({200.0f,200.0f,200.0f});

    skyboxObjects["stars"] = std::make_shared<moon::entities::SkyboxObject>(
        moon::utils::vkDefault::Paths{
            ExternalPath / "dependences/texture/skybox1/left.png",
            ExternalPath / "dependences/texture/skybox1/right.png",
            ExternalPath / "dependences/texture/skybox1/front.png",
            ExternalPath / "dependences/texture/skybox1/back.png",
            ExternalPath / "dependences/texture/skybox1/top.png",
            ExternalPath / "dependences/texture/skybox1/bottom.png"
    });
    skyboxObjects["stars"]->scale({200.0f,200.0f,200.0f});

    for(auto& [_,graph]: graphics){
        for(auto& [_,object]: objects){
            graph->bind(object->object());
        }
        for(auto& [_,object]: staticObjects){
            graph->bind(object->object());
        }
        for(auto& [_, object]: skyboxObjects){
            graph->bind(object->object());
        }
    }
}

void testScene::createLight()
{
    std::filesystem::path LIGHT_TEXTURE0  = ExternalPath / "dependences/texture/icon.PNG";
    std::filesystem::path LIGHT_TEXTURE1  = ExternalPath / "dependences/texture/light1.jpg";
    std::filesystem::path LIGHT_TEXTURE2  = ExternalPath / "dependences/texture/light2.jpg";
    std::filesystem::path LIGHT_TEXTURE3  = ExternalPath / "dependences/texture/light3.jpg";

    lightPoints["lightBox"] = std::make_shared<moon::entities::IsotropicLight>(
        moon::entities::IsotropicLight::Props{ moon::math::vec4(1.0f), 100.0f, true, false, 0.05f });
    groups["lightBox"]->add(lightPoints["lightBox"].get());

    using namespace moon::entities;

    auto addGroupLight = [&](const std::string& groupKey, std::shared_ptr<SpotLight> light) {
        groups[groupKey]->add(lightSources.emplace_back(std::move(light)).get());
        groupSpotLights[groupKey].push_back(static_cast<SpotLight*>(lightSources.back().get()));
    };

    addGroupLight("ufo0",      std::make_shared<SpotLight>(LIGHT_TEXTURE0, SpotLight::Props{ true, true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo1",      std::make_shared<SpotLight>(LIGHT_TEXTURE1, SpotLight::Props{ true, true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo2",      std::make_shared<SpotLight>(LIGHT_TEXTURE2, SpotLight::Props{ true, true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo3",      std::make_shared<SpotLight>(LIGHT_TEXTURE3, SpotLight::Props{ true, true, 0.05f, 10.0f, 0.3f }));

    addGroupLight("ufo_light_0", std::make_shared<SpotLight>(moon::math::vec4(1.00f, 0.65f, 0.20f, 1.00f), SpotLight::Props{  true,  true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo_light_1", std::make_shared<SpotLight>(moon::math::vec4(0.90f, 0.85f, 0.95f, 1.00f), SpotLight::Props{  true, false, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo_light_2", std::make_shared<SpotLight>(moon::math::vec4(0.90f, 0.85f, 0.75f, 1.00f), SpotLight::Props{  true,  true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo_light_3", std::make_shared<SpotLight>(moon::math::vec4(0.90f, 0.30f, 0.40f, 1.00f), SpotLight::Props{  true,  true, 0.05f, 10.0f, 0.3f }));
    addGroupLight("ufo_light_4", std::make_shared<SpotLight>(moon::math::vec4(0.20f, 0.50f, 0.95f, 1.00f), SpotLight::Props{  true,  true, 0.05f, 10.0f, 0.3f }));

    for (auto& [_, graph] : graphics) {
        for (const auto& light : lightPoints["lightBox"]->getLights()) {
            graph->bind(light);
        }
        for(auto& source: lightSources) {
            graph->bind(source->light());
        }
    }
}

void testScene::mouseEvent()
{
    const float sensitivity = mouse.control->sensitivity;

    const auto& cursorBuffer = mouse.cursor->read();
    const auto& cursorInfo = cursorBuffer.info;
    uint32_t primitiveNumber = cursorInfo.number;

    const auto xy = window.mousePose();
    if(mouse.control->pressed(GLFW_MOUSE_BUTTON_LEFT) || mouse.control->pressed(GLFW_MOUSE_BUTTON_RIGHT)){
        const auto delta = mouse.pose - xy;
        cameras["base"]->rotateX(sensitivity * delta[1]);
        cameras["base"]->rotateY(sensitivity * delta[0]);
    }
    if (mouse.control->pressed(GLFW_MOUSE_BUTTON_LEFT)) {
        const auto sizes = window.sizes();
        mouse.cursor->update(xy[0] / sizes[0], xy[1] / sizes[1]);
    }
    mouse.pose = xy;

    if(mouse.control->released(GLFW_MOUSE_BUTTON_LEFT)){
        for(auto& [key, object]: objects){
            moon::interfaces::Object* pObject = object->object();
            if (!pObject->comparePrimitive(primitiveNumber)) {
                continue;
            }
            if (controledObject.ptr == object.get()) {
                break;
            }
            auto newObject = dynamic_cast<moon::entities::BaseObject*>(object.get());
            if (!newObject) {
                continue;
            }
            auto oldObject = dynamic_cast<moon::entities::BaseObject*>(controledObject.ptr);
            if (oldObject) {
                oldObject->setOutlining(false);
            }
            controledObject.ptr = object.get();
            controledObject.name = key;
            newObject->setOutlining(controledObject.outlighting.enable, 0.03f, controledObject.outlighting.color);
            break;
        }
        requestUpdate();
    }

#ifdef SECOND_VIEW_WINDOW
    if(mouse.control->released(GLFW_MOUSE_BUTTON_RIGHT)) {
        if(cameras.count("view") > 0) {
            cameras["view"]->translation() = cameras["base"]->translation();
            cameras["view"]->rotation() = cameras["base"]->rotation();
            cameras["view"]->update();
        }
    }
#endif
}

void testScene::keyboardEvent()
{
    const float sensitivity = board->sensitivity;

    if (auto pBaseCamera = dynamic_cast<moon::entities::BaseCamera*>(cameras["base"].get()); pBaseCamera) {
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_A)) cameras["base"]->translate(-sensitivity * pBaseCamera->getViewMatrix()[0].dvec());
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_X)) cameras["base"]->translate(-sensitivity * pBaseCamera->getViewMatrix()[1].dvec());
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_W)) cameras["base"]->translate(-sensitivity * pBaseCamera->getViewMatrix()[2].dvec());
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_D)) cameras["base"]->translate( sensitivity * pBaseCamera->getViewMatrix()[0].dvec());
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_Z)) cameras["base"]->translate( sensitivity * pBaseCamera->getViewMatrix()[1].dvec());
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL) && board->pressed(GLFW_KEY_S)) cameras["base"]->translate( sensitivity * pBaseCamera->getViewMatrix()[2].dvec());
    }

    auto rotateControled = [this](const float& ang, const moon::math::vec3& ax) {
        if (bool foundInGroups = false; controledObject) {
            for (auto& [_, group] : groups) {
                if (group->find(controledObject)) {
                    group->rotate(ang, ax);
                    foundInGroups = true;
                }
            }
            if (!foundInGroups) controledObject->rotate(ang, ax);
        }
    };

    if (board->pressed(GLFW_KEY_KP_4)) rotateControled(moon::math::radians(0.5f), { 0.0f,0.0f,1.0f });
    if (board->pressed(GLFW_KEY_KP_6)) rotateControled(moon::math::radians(-0.5f), { 0.0f,0.0f,1.0f });
    if (board->pressed(GLFW_KEY_KP_8)) rotateControled(moon::math::radians(0.5f), { 1.0f,0.0f,0.0f });
    if (board->pressed(GLFW_KEY_KP_5)) rotateControled(moon::math::radians(-0.5f), { 1.0f,0.0f,0.0f });
    if (board->pressed(GLFW_KEY_KP_7)) rotateControled(moon::math::radians(0.5f), { 0.0f,1.0f,0.0f });
    if (board->pressed(GLFW_KEY_KP_9)) rotateControled(moon::math::radians(-0.5f), { 0.0f,1.0f,0.0f });

    auto translateControled = [this](const moon::math::vec3& tr){
        if(bool foundInGroups = false; controledObject){
            for(auto& [_,group]: groups){
                if(group->find(controledObject)){
                    group->translate(tr);
                    foundInGroups = true;
                }
            }
            if(!foundInGroups) controledObject->translate(tr);
        }
    };

    if(board->pressed(GLFW_KEY_LEFT))           translateControled(sensitivity * moon::math::vec3(-1.0f, 0.0f, 0.0f));
    if(board->pressed(GLFW_KEY_RIGHT))          translateControled(sensitivity * moon::math::vec3( 1.0f, 0.0f, 0.0f));
    if(board->pressed(GLFW_KEY_UP))             translateControled(sensitivity * moon::math::vec3( 0.0f, 1.0f, 0.0f));
    if(board->pressed(GLFW_KEY_DOWN))           translateControled(sensitivity * moon::math::vec3( 0.0f,-1.0f, 0.0f));
    if(board->pressed(GLFW_KEY_KP_ADD))         translateControled(sensitivity * moon::math::vec3( 0.0f, 0.0f, 1.0f));
    if(board->pressed(GLFW_KEY_KP_SUBTRACT))    translateControled(sensitivity * moon::math::vec3( 0.0f, 0.0f,-1.0f));

    if(board->released(GLFW_KEY_ESCAPE)) window.close();

    static uint32_t ufoCounter = 0;
    if(board->released(GLFW_KEY_N)) {
        std::random_device device;
        std::uniform_real_distribution dist(0.3f, 1.0f);
        moon::math::vec4 newColor = moon::math::vec4(dist(device), dist(device), dist(device), 1.0f);

		auto object = std::make_shared<moon::entities::BaseObject>(models["ufo"].get());
        object->rotate(moon::math::radians(90.0f), { 1.0f, 0.0f, 0.0f });

        moon::entities::SpotLight::Props lightProps{ true, true, 0.2f, 10.0f, 0.3f, 4.0f, moon::math::radians(90.0f), 1.0, 20.0f };
		auto light = std::make_shared<moon::entities::SpotLight>(newColor, lightProps);
		lightSources.push_back(light);

        auto group = std::make_shared<moon::transformational::Group>();
        group->translate(cameras["base"]->translation().im());
        group->add(light.get());
        group->add(object.get());

        auto objectKey = "new_ufo" + std::to_string(ufoCounter);
        objects[objectKey] = object;

		auto groupKey = "ufo_gr" + std::to_string(ufoCounter);
        groups[groupKey] = group;
        groupSpotLights[groupKey].push_back(light.get());

        for(auto& [_,graph]: graphics){
            graph->bind(light->light());
            graph->bind(object->object());
        }
		ufoCounter++;
    }

    if(board->released(GLFW_KEY_B)) {
        if(ufoCounter > 0 && app.deviceWaitIdle() == VK_SUCCESS) {
            ufoCounter--;
            auto objectKey = "new_ufo" + std::to_string(ufoCounter);
            for(auto& [_,graph]: graphics){
                graph->remove(objects[objectKey]->object());
                graph->remove(lightSources.back()->light());
            }
            lightSources.pop_back();
            objects.erase(objectKey);
        }
    }
}
