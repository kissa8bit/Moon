#ifndef TESTSCENE_H
#define TESTSCENE_H

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "scene.h"
#include "window.h"
#include "controller.h"
#include "controledObject.h"

#define SECOND_VIEW_WINDOW
#define IMGUI_GRAPHICS

namespace moon::interfaces { class Model;}
namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::deferredGraphics { class DeferredGraphics;}
namespace moon::transformational { class Light; class IsotropicLight; class Object; class Group; class Camera; class SkyboxObject;}
namespace moon::utils { class Cursor;}

class testScene : public scene
{
private:
    bool& framebufferResized;

    std::filesystem::path ExternalPath;
    moon::math::Vector<double,2> mousePos{0.0};

    float animationSpeed{1.0f};
    float frameTime{0.0f};

    moon::tests::Window window;
    moon::graphicsManager::GraphicsManager* app{nullptr};
    std::shared_ptr<controller> mouse{nullptr};
    std::shared_ptr<controller> board{nullptr};
    std::shared_ptr<moon::utils::Cursor> cursor{ nullptr };

    std::unordered_map<std::string, std::shared_ptr<moon::deferredGraphics::DeferredGraphics>> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>> models;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Camera>> cameras;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>> objects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>> staticObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>> skyboxObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Group>> groups;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::IsotropicLight>> lightPoints;
    std::vector<std::shared_ptr<moon::transformational::Light>> lightSources;

    moon::tests::ControledObject controledObject;

    void mouseEvent();
    void keyboardEvent();

    void create();
    void loadModels();
    void createLight();
    void createObjects();
    void requestUpdate();
    void makeGui();

public:
    testScene(moon::graphicsManager::GraphicsManager *app, GLFWwindow* window, uint32_t width, uint32_t height, const std::filesystem::path& ExternalPath, bool& framebufferResized);

    void resize(uint32_t width, uint32_t height) override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTSCENE_H
