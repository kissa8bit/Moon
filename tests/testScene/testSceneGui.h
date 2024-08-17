#ifndef TESTSCENEGUI_H
#define TESTSCENEGUI_H

#include "gui.h"

#include <controledObject.h>
#include <cameras.h>
#include <objects.h>
#include <deferredGraphics.h>

#include <memory>

namespace moon::tests::gui {

#define MOVE_VEC_DEF                                \
    static float move = 0.0f;                       \
    moon::math::Vector<float, 3> moveVec(0.0f);     \
    moveVec[ax] = 1.0f;                             \
    move = 0.0f;

template<int ax>
void sliderTransManipulator(
    transformational::Object& obj,
    const char* name,
    float min,
    float max,
    float width)
{
    MOVE_VEC_DEF

        ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat(name, &move, min, max)) {
        obj.translate(move * moveVec);
    }
    std::string text = " : " + std::to_string(obj.translation().im()[ax]);
    ImGui::SameLine(0.0, 10.0); ImGui::Text(text.c_str());
}


template<int ax>
void transManipulator(transformational::Object& obj, const char* name, float width = 100) {
    sliderTransManipulator<ax>(obj, name, -0.5f, 0.5f, width);
}

template<int ax>
void scaleSliderManipulator(
    transformational::Object& obj,
    const char* name,
    float min,
    float max,
    float width)
{
    MOVE_VEC_DEF

        ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat(name, &move, min, max)) {
        obj.scale(obj.scaling() + move * moveVec);
    }

    std::string text = " : " + std::to_string(obj.scaling()[ax]);
    ImGui::SameLine(0.0, 10.0); ImGui::Text(text.c_str());
}

template<int ax>
void scaleManipulator(transformational::Object& obj, const char* name, float width = 100) {
    scaleSliderManipulator<ax>(obj, name, -0.5f, 0.5f, width);
}

#undef MOVE_VEC_DEF

bool switcher(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, const std::string& name) {
    if (auto val = graphics->getEnable(name); ImGui::RadioButton(name.c_str(), val)) {
        graphics->setEnable(name, !val);
        return true;
    }
    return false;
}

bool switchers(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics) {
    bool framebufferResized = false;
    ImGui::BeginGroup();
    framebufferResized |= moon::tests::gui::switcher(graphics, "Bloom");
    framebufferResized |= moon::tests::gui::switcher(graphics, "Blur");
    framebufferResized |= moon::tests::gui::switcher(graphics, "Skybox");
    framebufferResized |= moon::tests::gui::switcher(graphics, "SSLR");
    framebufferResized |= moon::tests::gui::switcher(graphics, "SSAO");
    framebufferResized |= moon::tests::gui::switcher(graphics, "Shadow");
    framebufferResized |= moon::tests::gui::switcher(graphics, "Scattering");
    framebufferResized |= moon::tests::gui::switcher(graphics, "BoundingBox");
    framebufferResized |= moon::tests::gui::switcher(graphics, "TransparentLayer");
    ImGui::EndGroup();
    return framebufferResized;
}

bool setOutlighting(tests::ControledObject& obj, float width = 300.0f) {
    bool res = false;
    ImGui::BeginGroup();
    ImGui::Separator();
    auto& outlighting = obj.outlighting;
    auto& enable = outlighting.enable;
    auto& color = outlighting.color;
    if (ImGui::RadioButton("outlighting", enable)) {
        obj->setOutlining(enable = !enable);
        res = true;
    }
    ImGui::SetNextItemWidth(width);
    if (enable && ImGui::ColorEdit4("color", (float*)&color, ImGuiColorEditFlags_NoDragDrop)) {
        obj->setOutlining(enable, 0.03f, color);
        res = true;
    }
    ImGui::EndGroup();
    return res;
}

void rotationmManipulator(transformational::Object& obj, const moon::transformational::Camera* cam, float size = 100.0f) {
    float* rotation = (float*)&obj.rotation();
    float* camDir = (float*)&cam->getViewMatrix()[2].dvec();
    vec3 camdir(-camDir[0], -camDir[1], -camDir[2]);
    if (quat qu(rotation[0], rotation[1], rotation[2], rotation[3]);
        ImGui::gizmo3D("", qu, camdir, size, imguiGizmo::mode3Axes | imguiGizmo::sphereAtOrigin))
    {
        obj.rotation() = moon::math::Quaternion<float>(qu.w, qu.x, qu.y, qu.z);
        obj.update();
    }
}

}

#endif