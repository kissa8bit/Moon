#ifndef MOON_TEST_WINDOW_H
#define MOON_TEST_WINDOW_H

#include <filesystem>
#include <glfw3.h>

#include "linearAlgebra.h"

namespace moon::tests {

static bool resizeFlag = false;

class Window
{
private:
    math::vec2u extent{ 0 };
    GLFWwindow* window{ nullptr };

public:
    Window() = default;
    Window(const math::vec2u& extent, const std::filesystem::path& iconName = "");
    ~Window();

    operator GLFWwindow*() const;
    math::vec2u sizes() const;
    math::vec2d mousePose() const;
    float aspectRatio() const;
    bool isClosed() const;

    void resize();
    void close();
    bool& windowResized();
};

}
#endif // MOON_TEST_WINDOW_H