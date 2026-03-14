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

// #define TEST_RESOURCE_USAGE

#ifndef TEST_RESOURCE_USAGE
// #define SECOND_VIEW_WINDOW
#define IMGUI_GRAPHICS
#endif // !TEST_RESOURCE_USAGE

namespace moon::interfaces { class Model;}
namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::deferredGraphics { class DeferredGraphics;}
namespace moon::transformational { class Light; class Object; class Group; class Camera; class SkyboxObject;}
namespace moon::entities { class IsotropicLight; class SpotLight; class PointLight; }

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
    std::vector<moon::entities::PointLight*>                                            pointLightPtrs;
    float pointLightRadius{ 5.0f };
    float pointLightDrop{ 1.0f };

    // Point light volume (center + half-extents)
    float pointLightCx{ -27.0f }, pointLightCy{ -8.5f }, pointLightCz{ 15.0f };
    float pointLightHx{   5.0f }, pointLightHy{  3.5f }, pointLightHz{  5.0f };
    int   pointLightCount{ 30 };

    moon::tests::ControledObject controledObject;

    void mouseEvent();
    void keyboardEvent();

    void create();
    void createModels();
    void createLight();
    void createObjects();
    void spawnPointLights();
    void recreatePointLights();
    void requestUpdate();
#ifdef IMGUI_GRAPHICS
    void makeGui();
#endif

public:
    testScene(moon::graphicsManager::GraphicsManager& app, moon::tests::GlfwWindow& window, const std::filesystem::path& ExternalPath);
    ~testScene();

    void resize() override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
