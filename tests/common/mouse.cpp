#include "mouse.h"

namespace moon::tests {

Mouse::Mouse(GLFWwindow* window)
	: control(new controller(window, glfwGetMouseButton)), cursor(new moon::utils::Cursor) {}

Mouse::operator utils::Cursor*() const {
	return cursor.get();
}

}