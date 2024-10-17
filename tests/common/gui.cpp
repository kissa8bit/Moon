#include "gui.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif

namespace moon::tests::gui {

void printQuaternion(const moon::math::quat& quaternion) {
    float* rotation = (float*)&quaternion;
    ImGui::BeginGroup();
        ImGui::Text("rot_s : %s", std::to_string(rotation[0]).c_str());
        ImGui::Text("rot_x : %s", std::to_string(rotation[1]).c_str());
        ImGui::Text("rot_y : %s", std::to_string(rotation[2]).c_str());
        ImGui::Text("rot_z : %s", std::to_string(rotation[3]).c_str());
    ImGui::EndGroup();
}

void makeScreenshot(const std::string& screenshot, const std::vector<uint32_t>& data, uint32_t width, uint32_t height) {
    const uint32_t imageSize = height * width;
    const uint32_t imageBitesSize = 3 * imageSize;

    std::vector<uint8_t> jpg(imageBitesSize, 0);
    for (size_t pixel_index = 0; pixel_index < imageSize; pixel_index++) {
        const auto& pixel = data[pixel_index];
        jpg[3 * pixel_index + 0] = (pixel & 0x00ff0000) >> 16;
        jpg[3 * pixel_index + 1] = (pixel & 0x0000ff00) >> 8;
        jpg[3 * pixel_index + 2] = (pixel & 0x000000ff) >> 0;
    }
    auto filename = std::string("./") + std::string(screenshot.data()) + std::string(".jpg");
    stbi_write_jpg(filename.c_str(), width, height, 3, jpg.data(), 100);
}

void makeScreenshot(const char* name, const moon::graphicsManager::GraphicsManager& app) {
    static char screenshot[128] = "screenshot";
    ImGui::SetNextItemWidth(100.0f);
    ImGui::InputText(" ", screenshot, IM_ARRAYSIZE(screenshot));

    ImGui::SameLine(0.0, 10.0f);
    if (ImGui::Button(name)) {
        const auto imageExtent = app.getImageExtent();
        const auto screenshotImage = app.makeScreenshot();
        makeScreenshot(screenshot, screenshotImage, imageExtent.width, imageExtent.height);
    }
}

void fpsPlot(float currentFrameFPS, uint32_t points) {
    static std::vector<float> fps(points, 0.0f);
    float average = 0.0f, max = std::numeric_limits<float>::min(), min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < fps.size() - 1; i++) {
        average += (fps[i] = fps[i + 1]);
        max = std::max(max, fps[i]);
        min = std::min(min, fps[i]);
    }
    average += (fps[fps.size() - 1] = currentFrameFPS);
    average /= points;
    max = std::max(max, fps[fps.size() - 1]);
    min = std::min(min, fps[fps.size() - 1]);
    ImGui::PlotLines(("FPS:\n[" + std::to_string(min) + ",\n" + std::to_string(max) + "]").c_str(), fps.data(), fps.size(), 0, ("average = " + std::to_string(average)).c_str(), min, max, { 250.0f, 100.0f });
}

void setPoseInWindow(std::shared_ptr<moon::graphicsManager::GraphicsInterface> graphics) {
    auto position = graphics->getPositionInWindow();
    auto& viewOffset = position.offset;
    auto& viewExtent = position.size;
    bool update = false;
    ImGui::BeginGroup();
        ImGui::SetNextItemWidth(150.0f);
        update |= ImGui::SliderFloat("v_offset_x", &viewOffset[0], 0.0f, 1.0f);
        ImGui::SetNextItemWidth(150.0f);
        update |= ImGui::SliderFloat("v_offset_y", &viewOffset[1], 0.0f, 1.0f);
        ImGui::SetNextItemWidth(150.0f);
        update |= ImGui::SliderFloat("v_extent_x", &viewExtent[0], 0.0f, 1.0f);
        ImGui::SetNextItemWidth(150.0f);
        update |= ImGui::SliderFloat("v_extent_y", &viewExtent[1], 0.0f, 1.0f);
    ImGui::EndGroup();
    if (update) {
        graphics->setPositionInWindow({ viewOffset, viewExtent });
    }
}

bool radioButtonUpdate(const char* name, bool& flag) {
    if (ImGui::RadioButton(name, flag)) {
        flag = !flag;
        return true;
    }
    return false;
};

}