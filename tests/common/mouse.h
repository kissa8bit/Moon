#ifndef MOUSE_H
#define MOUSE_H

#include "cursor.h"
#include "controller.h"
#include "vector.h"

#include <memory>
#include <glfw3.h>

namespace moon::tests {

struct Mouse {
    math::Vector<double, 2> pose{ 0.0 };
    std::shared_ptr<controller> control{ nullptr };
    std::shared_ptr<utils::Cursor> cursor{ nullptr };

    Mouse() = default;
    Mouse(GLFWwindow* window);
    operator utils::Cursor*() const;
};

}

#endif