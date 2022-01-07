#ifndef TESTPOS_H
#define TESTPOS_H

#include <glfw3.h>

#include <filesystem>
#include <memory>

#include "scene.h"
#include "glfwWindow.h"
#include "controller.h"

#include <math/linearAlgebra.h>

#define IMGUI_GRAPHICS

namespace moon::graphicsManager { class GraphicsManager;}
namespace moon::imguiGraphics { class ImguiGraphics;}
namespace moon::rayTracingGraphics { class RayTracingGraphics;}
namespace cuda::rayTracing { struct Object; struct Camera; }

#include <cudaRayTracing/utils/devicep.h>

class testCuda : public scene
{
private:
    std::filesystem::path ExternalPath;
    moon::math::vec2d mousePos{0.0};

    float blitFactor = 1.0f;

    moon::graphicsManager::GraphicsManager& app;
    moon::tests::GlfwWindow& window;
    std::unique_ptr<cuda::rayTracing::Camera> hostcam;

    std::shared_ptr<controller> mouse;
    std::shared_ptr<controller> board;
    std::shared_ptr<moon::rayTracingGraphics::RayTracingGraphics> graphics;
#ifdef IMGUI_GRAPHICS
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;
#endif

    bool enableBB{true};
    bool enableBloom{true};

    cuda::rayTracing::Devicep<cuda::rayTracing::Camera> cam;
    std::unordered_map<std::string, std::unique_ptr<cuda::rayTracing::Object>> objects;

    void mouseEvent(float frameTime);
    void keyboardEvent(float frameTime);

    void create();
public:
    testCuda(moon::graphicsManager::GraphicsManager& app, moon::tests::GlfwWindow& window, const std::filesystem::path& ExternalPath);
    ~testCuda();

    void resize() override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;
};

#endif // TESTPOS_H
