#ifndef TESTSCENEGUI_H
#define TESTSCENEGUI_H

#include <memory>

#include "gui.h"

#include <controledObject.h>

#include <entities/baseCamera.h>
#include <entities/baseObject.h>
#include <transformationals/cameras.h>
#include <transformationals/objects.h>
#include <deferredGraphics/deferredGraphics.h>

namespace moon::tests::gui {

#define MOVE_VEC_DEF                                \
    static float move = 0.0f;                       \
    moon::math::vec3 moveVec(0.0f);                 \
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
    auto pBaseObject = dynamic_cast<entities::BaseObject*>(obj.ptr);
    if(!pBaseObject) return false;

    bool res = false;
    ImGui::BeginGroup();
    auto& outlighting = obj.outlighting;
    auto& enable = outlighting.enable;
    auto& color = outlighting.color;
    if (ImGui::RadioButton("outlighting", enable)) {
        pBaseObject->setOutlining(enable = !enable);
        res = true;
    }
    ImGui::SetNextItemWidth(width);
    if (enable && ImGui::ColorEdit4("outlighting color", (float*)&color, ImGuiColorEditFlags_NoDragDrop)) {
        pBaseObject->setOutlining(enable, 0.03f, color);
        res = true;
    }
    ImGui::EndGroup();
    return res;
}

void rotationmManipulator(transformational::Object& obj, const moon::entities::BaseCamera* cam, float size = 100.0f) {
    float* rotation = (float*)&obj.rotation();
    if(!cam) return;
    float* camDir = (float*)&cam->getViewMatrix()[2].dvec();
    vec3 camdir(-camDir[0], -camDir[1], -camDir[2]);
    if (quat qu(rotation[0], rotation[1], rotation[2], rotation[3]);
        ImGui::gizmo3D("", qu, camdir, size, imguiGizmo::mode3Axes | imguiGizmo::sphereAtOrigin))
    {
        obj.rotation() = moon::math::quat(qu.w, qu.x, qu.y, qu.z);
        obj.update();
    }
}

bool setColors(moon::transformational::Object* obecjt, float width = 300.0f) {
    auto pBaseObject = dynamic_cast<entities::BaseObject*>(obecjt);
    if (!pBaseObject) return false;

    static moon::math::vec4 constColor = { 0.0f };
    static moon::math::vec4 colorFactor = { 1.0f };
    ImGui::SetNextItemWidth(width);
    ImGui::ColorEdit4("const color", (float*)&constColor, ImGuiColorEditFlags_NoDragDrop);
    ImGui::SetNextItemWidth(width);
    ImGui::ColorEdit4("color factor", (float*)&colorFactor, ImGuiColorEditFlags_NoDragDrop);
    if (ImGui::Button("update")) {
        pBaseObject->setColor(constColor, colorFactor);
        return true;
    }
    return false;
}

bool setPlyMaterial(moon::models::PlyModel* model) {
    if (!model) return false;
    auto& material = model->material();
    static bool metallic = material.pbrWorkflows == moon::interfaces::Material::METALLIC_ROUGHNESS;
    bool update = false;
    if (ImGui::RadioButton("metallic roughbess", metallic)) {
        material.pbrWorkflows = moon::interfaces::Material::METALLIC_ROUGHNESS;
        metallic = true;
        update |= true;
    }
    if (ImGui::RadioButton("specular glossiness", !metallic)) {
        material.pbrWorkflows = moon::interfaces::Material::SPECULAR_GLOSSINESS;
        metallic = false;
        update |= true;
    }
    ImGui::SetNextItemWidth(150.0f);
    update |= ImGui::SliderFloat("metallic", &material.metallicRoughness.factor[moon::interfaces::Material::metallicIndex], 0.0f, 1.0f);
    ImGui::SetNextItemWidth(150.0f);
    update |= ImGui::SliderFloat("roughness", &material.metallicRoughness.factor[moon::interfaces::Material::roughnessIndex], 0.0f, 1.0f);
    return update;
}

bool graphicsProps(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, std::shared_ptr<utils::Cursor> cursor = nullptr) {
    if(!graphics) return false;

    ImGui::SetNextItemWidth(150.0f);
    if (float val = graphics->parameters().minAmbientFactor(); ImGui::SliderFloat("ambient", &val, 0.0f, 1.0f)) {
        graphics->parameters().minAmbientFactor() = val;
    }

    ImGui::SetNextItemWidth(150.0f);
    if (float val = graphics->parameters().blitFactor(); ImGui::SliderFloat("bloom factor", &val, 1.0f, 3.0f)) {
        graphics->parameters().blitFactor() = val;
    }

    if (cursor) {
        ImGui::SetNextItemWidth(150.0f);
        const moon::utils::CursorBuffer& cursorBuffer = cursor->read();
        float farBlurDepth = cursorBuffer.info.depth;
        ImGui::SliderFloat("far blur depth", &farBlurDepth, 0.0f, 1.0f);
        graphics->parameters().blurDepth() = graphics->getEnable("Blur") ? 1.02f * farBlurDepth : 1.0f;
    }

    if (bool val = graphics->parameters().scatteringRefraction(); ImGui::RadioButton("refraction of scattering", val)) {
        graphics->parameters().scatteringRefraction() = !val;
    }

    return true;
}

}

#endif