#ifndef GUI_H
#define GUI_H

#include <memory>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imguiGraphics.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#define IMGUIZMO_IMGUI_FOLDER
#include <imGuIZMO.quat/imGuIZMOquat.h>

#include <math/quaternion.h>
#include <graphicsManager/graphicsManager.h>

namespace moon::tests::gui {

void makeScreenshot(const char* name, const moon::graphicsManager::GraphicsManager& app);

void fpsPlot(float currentFrameFPS, uint32_t points = 100);

void printQuaternion(const moon::math::quat& quaternion);

void setPoseInWindow(std::shared_ptr<moon::graphicsManager::GraphicsInterface> graphics);

bool radioButtonUpdate(const char* name, bool& flag);

}

#endif // GUI_H