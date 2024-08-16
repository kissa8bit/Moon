#ifndef TESTPOS_H
#define TESTPOS_H

#include <glfw3.h>

#include <filesystem>
#include <memory>

#include "scene.h"
#include "window.h"
#include "controller.h"
#include "vector.h"

#define IMGUI_GRAPHICS

namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::rayTracingGraphics { class RayTracingGraphics;}
namespace cuda::rayTracing { struct Object; struct Camera; }

#include "utils/devicep.h"

class testCuda : public scene
{
private:
    std::filesystem::path ExternalPath;
    moon::math::Vector<double,2> mousePos{0.0};

    float focus = 0.049f;
    float blitFactor = 1.0f;
    std::string screenshot;

    moon::graphicsManager::GraphicsManager& app;
    moon::tests::Window& window;
    std::unique_ptr<cuda::rayTracing::Camera> hostcam;

    std::shared_ptr<controller> mouse;
    std::shared_ptr<controller> board;
    std::shared_ptr<moon::rayTracingGraphics::RayTracingGraphics> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    bool enableBB{true};
    bool primitivesBB{false};
    bool treeBB{true};
    bool onlyLeafsBB{false};
    bool enableBloom{true};

    cuda::rayTracing::Devicep<cuda::rayTracing::Camera> cam;
    std::unordered_map<std::string, std::unique_ptr<cuda::rayTracing::Object>> objects;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

    void create();
public:
    testCuda(moon::graphicsManager::GraphicsManager& app, moon::tests::Window& window, const std::filesystem::path& ExternalPath);
    ~testCuda();

    void resize() override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTPOS_H
