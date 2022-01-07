#include <chrono>
#include <stdexcept>
#include <utility>
#include <filesystem>

#include <graphicsManager/graphicsManager.h>
#include <utils/memory.h>

#include "glfwWindow.h"

#if defined(TESTCUDA)
    #include "testCuda.h"
#else
    #include "testScene.h"
#endif

VkPhysicalDeviceFeatures physicalDeviceFeatures(){
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.independentBlend = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;
    deviceFeatures.imageCubeArray = VK_TRUE;
    deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    deviceFeatures.fillModeNonSolid = VK_TRUE;
    return deviceFeatures;
}

std::pair<uint32_t, uint32_t> resize(moon::tests::GlfwWindow& window, moon::graphicsManager::GraphicsManager& app, scene& testScene) {
    window.resize();
    app.reset((moon::utils::Window*)&window);
    testScene.resize();
    return { window.sizes()[0], window.sizes()[1] };
}

using clk = std::chrono::high_resolution_clock;
template<typename type>
type period(clk::time_point time){
    return std::chrono::duration<type, std::chrono::seconds::period>(clk::now() - time).count();
}

int main()
{
    uint32_t WIDTH = 1024;
    uint32_t HEIGHT = 720;
    uint32_t resCount = 2;
    uint32_t imageCount = 3;

    const std::filesystem::path ExternalPath = ROOT_PATH;

    moon::tests::GlfwWindow window({WIDTH, HEIGHT}, ExternalPath / "dependences/texture/icon.PNG");

    moon::graphicsManager::GraphicsManager app((moon::utils::Window*)&window, imageCount, resCount, physicalDeviceFeatures());

#if defined(TESTCUDA)
    testCuda testScene(app, window, ExternalPath);
#else
    testScene testScene(app, window, ExternalPath);
#endif

#ifndef DEBUG_PRINT_DISABLE
    moon::utils::Memory::instance().status();
#endif

    for(float time = 1.0f; !window.isClosed();){
        if(auto start = clk::now(); app.checkNextFrame() != VK_ERROR_OUT_OF_DATE_KHR) {
            testScene.updateFrame(app.getResourceIndex(), time);

            if (VkResult result = app.drawFrame(); result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.windowResized()){
                std::tie(WIDTH, HEIGHT) = resize(window, app, testScene);
            } else if(result) {
                throw std::runtime_error("failed to with " + std::to_string(result));
            }
            time = period<float>(start);
        }
    }

    moon::utils::debug::checkResult(app.deviceWaitIdle(), "in file " + std::string(__FILE__) + ", line " + std::to_string(__LINE__));

    return 0;
}
