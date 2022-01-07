#include "glfwWindow.h"

#include <stb_image.h>

namespace moon::tests {

GlfwWindow::GlfwWindow(const math::vec2u& extent, const std::filesystem::path& iconName)
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

GlfwWindow::~GlfwWindow() {
    glfwDestroyWindow(window);
    glfwTerminate();
}

GlfwWindow::operator GLFWwindow* () const {
	return window;
}

math::vec2u GlfwWindow::sizes() const {
	return extent;
}

bool& GlfwWindow::windowResized() {
    return resizeFlag;
}

void GlfwWindow::resize() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width * height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }
    extent = math::vec2u(width, height);

    resizeFlag = false;
}

float GlfwWindow::aspectRatio() const {
	return (float)extent[0] / (float)extent[1];
}

void GlfwWindow::close() {
	glfwSetWindowShouldClose(window, GLFW_TRUE);
}

bool GlfwWindow::isClosed() const {
    return glfwWindowShouldClose(window);
}

math::vec2d GlfwWindow::mousePose() const {
	double x = 0, y = 0;
	glfwGetCursorPos(window, &x, &y);
	return { x,y };
}

utils::Window::FramebufferSize GlfwWindow::getFramebufferSize() const {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    return utils::Window::FramebufferSize{ static_cast<uint32_t>(width) , static_cast<uint32_t>(height) };
}

VkResult GlfwWindow::createSurface(VkInstance instance, VkSurfaceKHR* surface) const {
    return glfwCreateWindowSurface(instance, window, nullptr, surface);
}

std::vector<const char*> GlfwWindow::getExtensions() const {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
}

}