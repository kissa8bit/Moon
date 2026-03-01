#ifndef TESTSCENEGUI_H
#define TESTSCENEGUI_H

#include <memory>

#include "gui.h"

#include <controledObject.h>

#include <entities/baseCamera.h>
#include <entities/baseObject.h>
#include <entities/spotLight.h>
#include <transformationals/cameras.h>
#include <transformationals/objects.h>
#include <deferredGraphics/deferredGraphics.h>

namespace moon::tests::gui {

// DragFloat: drag to move, Ctrl+click to type exact value.
template<int ax, typename T>
void transManipulator(T& obj, const char* name, float width = 100) {
    float pos = obj.translation().im()[ax];
    ImGui::SetNextItemWidth(width);
    if (ImGui::DragFloat(name, &pos, 0.05f, 0.0f, 0.0f, "%.2f")) {
        moon::math::vec3 delta(0.0f);
        delta[ax] = pos - obj.translation().im()[ax];
        obj.translate(delta);
    }
}

template<int ax>
void scaleManipulator(transformational::Object& obj, const char* name, float width = 100) {
    moon::math::vec3 newScale = obj.scaling();
    ImGui::SetNextItemWidth(width);
    if (ImGui::DragFloat(name, &newScale[ax], 0.01f, 0.0f, 0.0f, "%.3f")) {
        obj.scale(newScale);
    }
}

inline bool switcher(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, const moon::workflows::ParameterName& name) {
    if (auto val = graphics->getEnable(name); ImGui::RadioButton(name.get().c_str(), val)) {
        graphics->setEnable(name, !val);
        return true;
    }
    return false;
}

inline bool switchers(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics) {
    bool framebufferResized = false;
    ImGui::BeginGroup();
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::Bloom::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::Blur::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::Skybox::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::SSLR::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::SSAO::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::Shadow::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::Scattering::param);
    framebufferResized |= moon::tests::gui::switcher(graphics, moon::deferredGraphics::Names::BoundingBox::param);
    ImGui::EndGroup();
    return framebufferResized;
}

inline bool setOutlighting(tests::ControledObject& obj, float width = 300.0f) {
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

template<typename T>
void rotationmManipulator(T& obj, const moon::entities::BaseCamera* cam, float size = 100.0f) {
    if (!cam) return;

	const moon::math::mat3 camRotMat = cam->getViewMatrix().extract(3, 3);
    const auto camRotation = moon::math::convert(camRotMat);
    const auto localRotation = camRotation * obj.rotation();

    float* p_rotation = (float*)&localRotation;
    if (quat qu(p_rotation[0], p_rotation[1], p_rotation[2], p_rotation[3]);
        ImGui::gizmo3D("", qu, size, imguiGizmo::mode3Axes | imguiGizmo::sphereAtOrigin))
    {
        obj.rotation() = camRotation.inverted() * moon::math::quat(qu.w, qu.x, qu.y, qu.z);
        obj.update();
    }
}

inline bool setColors(moon::transformational::Object* object, float width = 300.0f) {
    bool res = false;

    auto pBaseObject = dynamic_cast<entities::BaseObject*>(object);
    if (!pBaseObject) return res;

    static moon::math::vec4 constColor = { 0.0f };
    static moon::math::vec4 colorFactor = { 1.0f };
    ImGui::SetNextItemWidth(width);
    res |= ImGui::ColorEdit4("const color", (float*)&constColor, ImGuiColorEditFlags_NoDragDrop);
    ImGui::SetNextItemWidth(width);
    res |= ImGui::ColorEdit4("color factor", (float*)&colorFactor, ImGuiColorEditFlags_NoDragDrop);

    if(res){
        pBaseObject->setColor(constColor, colorFactor);
    }

    return res;
}

inline bool setPlyMaterial(moon::models::PlyModel* model) {
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

template<typename T>
bool spotLightSliders(T& light, int index, float width = 150.0f) {
	bool res = false;
    ImGui::PushID(index);
    float drop = light.getDrop();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("drop",         &drop,         0.0f,  1.0f))  light.setDrop(drop);
    float power = light.getPower();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("power",        &power,        0.0f, 100.0f)) light.setPower(power);
    float inner = light.getInnerFraction();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("innerFraction",&inner,        0.0f,  1.0f))  light.setInnerFraction(inner);
    float exp = light.getExponent();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("exponent",     &exp,          0.0f, 20.0f))  light.setExponent(exp);
    bool shadow = light.getEnableShadow();
    if (ImGui::RadioButton("shadow", shadow)) {
        light.setEnableShadow(!shadow);
        res = true;
    }
    ImGui::SameLine();
    bool scattering = light.getEnableScattering();
    if (ImGui::RadioButton("scattering", scattering)) {
        light.setEnableScattering(!scattering);
        res = true;
    }
    ImGui::PopID();
    return res;
}

inline void spotLightProjectionSliders(moon::entities::SpotLight& light, int index, float width = 150.0f) {
    ImGui::PushID(index);
    float fovDeg = light.getFov() / moon::math::radians(1.0f);
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("fov",    &fovDeg,          5.0f, 175.0f)) light.setFov(moon::math::radians(fovDeg));
    float aspect = light.getAspect();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("aspect", &aspect,           0.1f,   4.0f)) light.setAspect(aspect);
    float farPlane = light.getFar();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("far",    &farPlane,         0.1f, 100.0f)) light.setFar(farPlane);
    ImGui::PopID();
}

inline void scatteringProps(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, float width = 150.0f) {
    if (!graphics) return;
    auto& params = graphics->scatteringWorkflowParams();
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("density", &params.density, 0.0f, 1.0f)){
		graphics->requestUpdate(moon::deferredGraphics::Names::Scattering::name);
    }
    ImGui::SetNextItemWidth(width);
    if(ImGui::SliderInt("steps", &params.steps, 2, 200)){
        graphics->requestUpdate(moon::deferredGraphics::Names::Scattering::name);
	}
}

inline bool graphicsProps(std::shared_ptr<moon::deferredGraphics::DeferredGraphics> graphics, std::shared_ptr<utils::Cursor> cursor = nullptr) {
    if(!graphics) return false;

    ImGui::SetNextItemWidth(150.0f);
    if (float val = graphics->parameters().minAmbientFactor(); ImGui::SliderFloat("ambient", &val, 0.0f, 1.0f)) {
        graphics->parameters().minAmbientFactor() = val;
    }

    ImGui::SetNextItemWidth(150.0f);
    if (float val = graphics->parameters().blitFactor(); ImGui::SliderFloat("bloom factor", &val, 1.0f, 3.0f)) {
        graphics->parameters().blitFactor() = val;
    }

    static bool manualBlurDepth = false;
    static float farBlurDepth = 1.0f;
    auto depthTransform = [](float val) {
        constexpr float deg = 0.1f;
		return std::pow(val, deg) + std::numeric_limits<float>::epsilon();
	};
    if (ImGui::RadioButton("manual blur depth", manualBlurDepth)) {
        manualBlurDepth = !manualBlurDepth;
        graphics->parameters().blurDepth() = manualBlurDepth ? depthTransform(farBlurDepth) : 0.0f;
    }
    if (manualBlurDepth) {
        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::SliderFloat("blur depth", &farBlurDepth, 0.0f, 1.0f)) {
            graphics->parameters().blurDepth() = depthTransform(farBlurDepth);
        }
    }

    return true;
}

}

#endif