#include "window.h"

#include <stb_image.h>

namespace moon::tests {

Window::Window(const math::Vector<uint32_t, 2>& extent, const std::filesystem::path& iconName)
	: extent(extent)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(extent[0], extent[1], "Vulkan render", nullptr, nullptr);
    glfwSetWindowUserPointer(window, nullptr);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int){resizeFlag = true;});

    if (iconName.string().size() > 0) {
        int width, height, comp;
        stbi_uc* img = stbi_load(iconName.string().c_str(), &width, &height, &comp, 0);
        GLFWimage images{ width,height,img };
        glfwSetWindowIcon(window, 1, &images);
    }
}

Window::~Window() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

Window::operator GLFWwindow* () const {
	return window;
}

math::Vector<uint32_t, 2> Window::sizes() const {
	return extent;
}

bool& Window::windowResized() {
    return resizeFlag;
}

void Window::resize() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    extent = math::Vector<uint32_t, 2>(width, height);

    resizeFlag = false;
}

float Window::aspectRatio() const {
	return (float)extent[0] / (float)extent[1];
}

void Window::close() {
	glfwSetWindowShouldClose(window, GLFW_TRUE);
}

bool Window::isClosed() const {
    return glfwWindowShouldClose(window);
}

math::Vector<double, 2> Window::mousePose() const {
	double x = 0, y = 0;
	glfwGetCursorPos(window, &x, &y);
	return { x,y };
}

}