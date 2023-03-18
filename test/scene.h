#ifndef SCENE_H
#define SCENE_H

#include <glfw3.h>
#include <string>
#include <vector>

class deferredGraphics;
struct gltfModel;
class graphicsManager;
class spotLight;
class isotropicLight;
class object;
class group;
class camera;
class skyboxObject;

void scrol(GLFWwindow* window, double xoffset, double yoffset);

class scene
{
private:
    std::string ExternalPath;
    uint32_t    WIDTH;
    uint32_t    HEIGHT;

    float       globalTime = 0.0f;
    float       timeScale = 1.0f;
    float       minAmbientFactor = 0.05f;

    double      xMpos, yMpos;
    double      angx=0.0, angy=0.0;
    bool        mouse1Stage = 0;
    bool        backRStage = 0;
    bool        backTStage = 0;
    bool        backYStage = 0;
    bool        backNStage = 0;
    bool        backBStage = 0;
    bool        backOStage = 0;
    bool        backIStage = 0;
    bool        backGStage = 0;
    bool        backHStage = 0;

    uint32_t    controledGroup = 0;
    uint32_t    lightPointer = 10;

    std::string ZERO_TEXTURE;
    std::string ZERO_TEXTURE_WHITE;

    camera*                                         cameras;
    skyboxObject*                                   skyboxObject1;
    skyboxObject*                                   skyboxObject2;

    std::vector<std::vector<gltfModel   *>>         gltfModel;
    std::vector<object                  *>          object3D;
    std::vector<spotLight               *>          lightSource;
    std::vector<isotropicLight          *>          lightPoint;
    std::vector<group                   *>          groups;

    graphicsManager*            app;
    deferredGraphics*  graphics;

    void mouseEvent(GLFWwindow* window, float frameTime);
    void keyboardEvent(GLFWwindow* window, float frameTime);
    void updates(float frameTime);

    void loadModels();
    void createLight();
    void createObjects();
public:
    scene(graphicsManager *app, deferredGraphics* graphics, std::string ExternalPath);
    void createScene(uint32_t WIDTH, uint32_t HEIGHT, camera* cameraObject);
    void updateFrame(GLFWwindow* window, uint32_t frameNumber, float frameTime, uint32_t WIDTH, uint32_t HEIGHT);
    void destroyScene();

    bool framebufferResized = false;
};

#endif // SCENE_H