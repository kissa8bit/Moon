#ifndef TESTSCENE_H
#define TESTSCENE_H

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "scene.h"
#include "glfwWindow.h"
#include "controller.h"
#include "controledObject.h"
#include "mouse.h"

// #define SECOND_VIEW_WINDOW
#define IMGUI_GRAPHICS

namespace moon::interfaces { class Model;}
namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::deferredGraphics { class DeferredGraphics;}
namespace moon::transformational { class Light; class Object; class Group; class Camera; class SkyboxObject;}
namespace moon::entities { class IsotropicLight; class SpotLight; }

class testScene : public scene
{
private:
    std::filesystem::path ExternalPath;

    moon::graphicsManager::GraphicsManager& app;
    moon::tests::GlfwWindow& window;
    moon::tests::Mouse mouse;
    std::shared_ptr<controller> board{ nullptr };

    float animationSpeed{1.0f};
    float frameTime{0.0f};

    std::unordered_map<std::string, std::shared_ptr<moon::deferredGraphics::DeferredGraphics>> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>>           models;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Camera>>    cameras;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>    objects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>    staticObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>    skyboxObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Group>>     groups;
    std::unordered_map<std::string, std::shared_ptr<moon::entities::IsotropicLight>>    lightPoints;
    std::vector<std::shared_ptr<moon::transformational::Light>> lightSources;
    std::unordered_map<std::string, std::vector<moon::entities::SpotLight*>>            groupSpotLights;

    moon::tests::ControledObject controledObject;

    void mouseEvent();
    void keyboardEvent();

    void create();
    void createModels();
    void createLight();
    void createObjects();
    void requestUpdate();
    void makeGui();

public:
    testScene(moon::graphicsManager::GraphicsManager& app, moon::tests::GlfwWindow& window, const std::filesystem::path& ExternalPath);
    ~testScene();

    void resize() override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
