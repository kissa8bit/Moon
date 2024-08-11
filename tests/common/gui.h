#ifndef GUI_H
#define GUI_H

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imguiGraphics.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#define IMGUIZMO_IMGUI_FOLDER
#include <imGuIZMO.quat/imGuIZMOquat.h>

#include <controledObject.h>
#include <quaternion.h>
#include <cameras.h>
#include <objects.h>
#include <graphicsManager.h>
#include <deferredGraphics.h>

#include <memory>

namespace moon::tests::gui {

bool setOutlighting(tests::ControledObject& obj, float width = 300.0f);

void printQuaternion(const moon::math::Quaternion<float>& quaternion);

void rotationmManipulator(transformational::Object& obj, const moon::transformational::Camera* cam, float size = 100.0f);

template<int ax>
void transManipulator(transformational::Object& obj, const char* name, float width = 100.0f);

template<int ax>
void scaleManipulator(transformational::Object& obj, const char* name, float width = 100.0f);

void makeScreenshot(const char* name, const moon::graphicsManager::GraphicsManager& app);

void fpsPlot(float currentFrameFPS, uint32_t points = 100);

bool switcher(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, const std::string& name);

bool switchers(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics);

void setPoseInWindow(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics);

}

#endif // GUI_H