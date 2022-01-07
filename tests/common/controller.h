#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <stdint.h>
#include <unordered_map>

struct GLFWwindow;

class controller
{
private:
    std::unordered_map<uint32_t, bool> keysMap;
    GLFWwindow* window{nullptr};
    int (*glfwGetFunction)(GLFWwindow*,int){nullptr};

public:
    float sensitivity{0.1f};

public:
    controller() = default;
    controller(GLFWwindow* window, int (*glfwGetFunction)(GLFWwindow*,int));
    bool pressed(uint32_t key);
    bool released(uint32_t key);
};

#endif // CONTROLLER_H
