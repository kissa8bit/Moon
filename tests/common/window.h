#ifndef MOON_TEST_WINDOW_H
#define MOON_TEST_WINDOW_H

#include <glfw3.h>

#include "vector.h"

namespace moon::tests {

class Window
{
private:
    math::Vector<uint32_t, 2> extent{ 0 };
    GLFWwindow* window{ nullptr };

public:
    Window() = default;
    Window(GLFWwindow* window, const math::Vector<uint32_t, 2>& extent);

    operator GLFWwindow*() const;
    math::Vector<uint32_t, 2> sizes() const;
    void resizes(const math::Vector<uint32_t, 2>& size);
    float aspectRatio() const;

    void close();
    math::Vector<double, 2> mousePose() const;
};

}
#endif // MOON_TEST_WINDOW_H