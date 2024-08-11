#ifndef MOON_TEST_WINDOW_H
#define MOON_TEST_WINDOW_H

#include <filesystem>
#include <glfw3.h>

#include "vector.h"

namespace moon::tests {

static bool resizeFlag = false;

class Window
{
private:
    math::Vector<uint32_t, 2> extent{ 0 };
    GLFWwindow* window{ nullptr };

public:
    Window() = default;
    Window(const math::Vector<uint32_t, 2>& extent, const std::filesystem::path& iconName = "");
    ~Window();

    operator GLFWwindow*() const;
    math::Vector<uint32_t, 2> sizes() const;
    math::Vector<double, 2> mousePose() const;
    float aspectRatio() const;
    bool isClosed() const;

    void resize();
    void close();
    bool& windowResized();
};

}
#endif // MOON_TEST_WINDOW_H