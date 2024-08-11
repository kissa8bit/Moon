#include "window.h"

namespace moon::tests {

Window::Window(GLFWwindow* window, const math::Vector<uint32_t, 2>& extent)
	: window(window), extent(extent)
{}

Window::operator GLFWwindow* () const {
	return window;
}

math::Vector<uint32_t, 2> Window::sizes() const {
	return extent;
}

void Window::resizes(const math::Vector<uint32_t, 2>& size) {
	extent = size;
}

float Window::aspectRatio() const {
	return (float)extent[0] / (float)extent[1];
}

void Window::close() {
	glfwSetWindowShouldClose(window, GLFW_TRUE);
}

math::Vector<double, 2> Window::mousePose() const {
	double x = 0, y = 0;
	glfwGetCursorPos(window, &x, &y);
	return { x,y };
}

}