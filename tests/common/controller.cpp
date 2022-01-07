#include "controller.h"
#include "glfw3.h"

controller::controller(GLFWwindow* window, int (*glfwGetFunction)(GLFWwindow*,int)) : window(window), glfwGetFunction(glfwGetFunction){}

bool controller::pressed(uint32_t key){
    bool res = glfwGetFunction(window,key) == GLFW_PRESS;
    return res;
}

bool controller::released(uint32_t key){
    bool res = keysMap.count(key) > 0 && keysMap[key] == GLFW_PRESS && glfwGetFunction(window,key) == GLFW_RELEASE;
    keysMap[key] = glfwGetFunction(window,key);
    return res;
}

