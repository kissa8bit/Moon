#ifndef MOUSE_H
#define MOUSE_H

#include <memory>
#include <glfw3.h>

#include <utils/cursor.h>

#include <math/linearAlgebra.h>

#include "controller.h"

namespace moon::tests {

struct Mouse {
    math::vec2d pose{ 0.0 };
    std::shared_ptr<controller> control{ nullptr };
    std::shared_ptr<utils::Cursor> cursor{ nullptr };

    Mouse() = default;
    Mouse(GLFWwindow* window);
    operator utils::Cursor*() const;
};

}

#endif