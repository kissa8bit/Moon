#ifndef MOON_TEST_WINDOW_H
#define MOON_TEST_WINDOW_H

#include <filesystem>
#include <vulkan.h>
#include <glfw3.h>

#include <math/linearAlgebra.h>
#include <utils/window.h>

namespace moon::tests {

static bool resizeFlag = false;

class GlfwWindow : utils::Window
{
private:
    math::vec2u extent{ 0 };
    GLFWwindow* window{ nullptr };

public:
    GlfwWindow() = default;
    GlfwWindow(const math::vec2u& extent, const std::filesystem::path& iconName = "");
    ~GlfwWindow();

    operator GLFWwindow*() const;
    math::vec2u sizes() const;
    math::vec2d mousePose() const;
    float aspectRatio() const;
    bool isClosed() const;

    void resize();
    void close();
    bool& windowResized();

    utils::Window::FramebufferSize getFramebufferSize() const override;
    VkResult createSurface(VkInstance instance, VkSurfaceKHR* surface) const override;
    std::vector<const char*> getExtensions() const override;
};

}
#endif // MOON_TEST_WINDOW_H