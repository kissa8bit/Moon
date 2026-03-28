#ifndef SCENE_EDITOR_H
#define SCENE_EDITOR_H

#include <filesystem>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <chrono>

#include "scene.h"
#include "glfwWindow.h"
#include "controller.h"
#include "mouse.h"

#include <math/linearAlgebra.h>
#include <math/quaternion.h>

namespace moon::interfaces { class Model; class Object; }
namespace moon::graphicsManager { class GraphicsManager; }
namespace moon::imguiGraphics { class ImguiGraphics; }
namespace moon::deferredGraphics { class DeferredGraphics; }
namespace moon::transformational { class Transformational; class Light; class Object; class Group; class Camera; }
namespace moon::entities { class SpotLight; class PointLight; class DirectionalLight; class IsotropicLight; class BaseObject; class BaseCamera; }

// ─────────────────────────────────────────────────────────────────────────────
// Scene node type tag
// ─────────────────────────────────────────────────────────────────────────────
enum class NodeKind { Object, SkyboxObject, Camera, Light, Group };

struct HierarchyNode {
    NodeKind                                  kind{ NodeKind::Object };
    std::string                               name;
    moon::transformational::Transformational* ptr{ nullptr };
};

// ─────────────────────────────────────────────────────────────────────────────
// Editor state shared between the scene and GUI panels
// ─────────────────────────────────────────────────────────────────────────────
struct EditorState {
    // Primary selection (for inspector; always an Object or nullptr)
    moon::transformational::Object* selectedObject{ nullptr };
    std::string selectedName;

    // Multi-selection set (any Transformational*)
    std::unordered_set<moon::transformational::Transformational*> selection;

    // Per-node visibility (keyed by name, works for all node types)
    std::unordered_map<std::string, bool> nodeVisible;

    // Locked nodes: can't be moved or grouped
    std::unordered_set<std::string> lockedNodes;

    // Saved light power for enable/disable toggle (SpotLight/PointLight)
    std::unordered_map<std::string, float> savedLightPower;

    // Panel visibility toggles (controlled via View menu)
    bool showHierarchy{ true };
    bool showInspector{ true };
    bool showPerformance{ true };

    // Add Object dialog state
    bool showAddObjectDialog{ false };

    // Deferred spawn request (processed at start of next frame, outside render pass)
    struct PendingObjectSpawn {
        std::filesystem::path filePath;
        float scale{ 1.0f };
        bool animated{ false };
        bool pending{ false };
    };
    PendingObjectSpawn pendingSpawn;

    // Deferred spawn requests (processed at start of next frame, outside render pass)
    enum class SpawnType { None, PointLight, SpotLight, DirectionalLight, IsotropicLight, Camera, Object };
    SpawnType pendingSpawnType{ SpawnType::None };
    std::string pendingSpawnModel; // for SpawnType::Object

    // Hierarchy search filter
    char hierarchyFilter[128]{};

    // Outline highlight for selected objects
    bool outlineEnabled{ true };
    moon::math::vec4 outlineColor{ 0.0f, 1.0f, 0.2f, 1.0f };

    // Render resolution scale (applied on top of window size)
    float resolutionScale{ 1.0f };
    bool  pendingResolutionChange{ false };

    // Scene file path (for save/load)
    char sceneFilePath[512]{};
    bool showLoadSceneDialog{ false };
    bool showSaveAsDialog{ false };
};

// ─────────────────────────────────────────────────────────────────────────────
// Scene editor – a self-contained scene with multi-panel ImGui UI
// ─────────────────────────────────────────────────────────────────────────────
class sceneEditor : public scene
{
public:
    // Editor group: parallel to engine Group, holds the child list we can iterate
    struct EditorGroup {
        moon::transformational::Group* group{ nullptr }; // non-owning (owned by groups map)
        std::vector<HierarchyNode>     children;
        std::shared_ptr<moon::entities::BaseObject> pivotCube; // pivot marker at group origin
        moon::transformational::Transformational* lightEntity{ nullptr }; // for atomic light units
    };

private:
    std::filesystem::path ExternalPath;

    moon::graphicsManager::GraphicsManager& app;
    moon::tests::GlfwWindow& window;
    moon::tests::Mouse mouse;
    std::shared_ptr<controller> board{ nullptr };

    float animationSpeed{ 1.0f };
    float frameTime{ 0.0f };

    std::unordered_map<std::string, std::shared_ptr<moon::deferredGraphics::DeferredGraphics>> graphics;
    std::shared_ptr<moon::imguiGraphics::ImguiGraphics> gui;

    std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>>           models;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Camera>>    cameras;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>    objects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>    skyboxObjects;
    std::unordered_map<std::string, std::shared_ptr<moon::transformational::Group>>     groups;

    std::shared_ptr<moon::entities::DirectionalLight>                               sunLight;
    std::vector<std::shared_ptr<moon::transformational::Light>>                     lightSources;
    std::unordered_map<std::string, std::vector<moon::entities::SpotLight*>>        groupSpotLights;
    std::vector<moon::entities::PointLight*>                                        pointLightPtrs;

    EditorState editorState;

    // ── Hierarchy tracking ────────────────────────────────────────────────────
    std::vector<HierarchyNode>                       hierarchyRoots;    // top-level items
    std::unordered_map<std::string, EditorGroup>     editorGroups;      // user-created groups
    std::unordered_map<moon::transformational::Transformational*, std::string> nodeNames; // ptr → name
    int groupNameCounter{ 0 };
    int objectNameCounter{ 0 };
    int spotLightNameCounter{ 0 };
    int dirLightNameCounter{ 0 };
    int isoLightNameCounter{ 0 };
    int cameraNameCounter{ 0 };

    // Marker cube → owning entity for viewport picking (pivot cubes, light markers, camera markers)
    std::unordered_map<moon::interfaces::Object*, std::pair<moon::transformational::Transformational*, std::string>> markerOwners;

    // Model path tracking for scene save/load (model key → relative path from ExternalPath)
    std::unordered_map<std::string, std::string> modelPaths;
    // Object → model key association for save
    std::unordered_map<std::string, std::string> objectModelKeys;
    // Skybox texture paths (name → relative paths from ExternalPath)
    std::unordered_map<std::string, std::vector<std::string>> skyboxTexturePaths;

    std::string activeCameraName{ "base" };

    // Double-click detection for viewport picking
    std::chrono::steady_clock::time_point lastClickTime{};
    moon::math::vec2d lastClickPos{ 0.0 };
    static constexpr double doubleClickMaxSeconds = 0.4;
    static constexpr double doubleClickMaxDist = 10.0;

    // Internals
    void create();
    void createModels();
    void createObjects();
    void createLights();
    void loadSceneFromJson(const std::filesystem::path& jsonPath);

    void mouseEvent();
    void keyboardEvent();
    void makeGui();

    void registerNode(NodeKind kind, const std::string& name,
                      moon::transformational::Transformational* ptr);
    void groupSelected();
    void disbandGroup(const std::string& grpName);
    void toggleSelection(moon::transformational::Transformational* item,
                         const std::string& name, bool ctrlHeld);
    void clearSelection();
    void syncOutlines();

    void setupEngineGroups();   // mirrors pre-existing engine groups into hierarchy
    void selectObject(moon::transformational::Object* obj, const std::string& name);
    void spawnPointLight();
    void spawnObject(const std::string& modelName);
    void spawnObjectFromFile(const std::filesystem::path& filePath, float scale, bool animated);
    void spawnSpotLight();
    void spawnDirectionalLight();
    void spawnIsotropicLight();
    void spawnCamera();
    void switchToCamera(const std::string& camName);
    void focusOnSelected();

public:
    sceneEditor(moon::graphicsManager::GraphicsManager& app,
                moon::tests::GlfwWindow& window,
                const std::filesystem::path& ExternalPath);
    ~sceneEditor();

    void resize() override;
    void updateFrame(uint32_t frameNumber, float frameTime) override;

    // Accessors used by GUI panels
    moon::deferredGraphics::DeferredGraphics* baseGraphics() const;
    moon::transformational::Camera*           activeCamera() const;
    EditorState&                              state() { return editorState; }
    float                                     fps()   const { return frameTime > 0.0f ? 1.0f / frameTime : 0.0f; }

    const std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>& getObjects()       const { return objects; }
    const std::unordered_map<std::string, std::shared_ptr<moon::transformational::Object>>& getSkyboxObjects() const { return skyboxObjects; }
    const std::unordered_map<std::string, std::shared_ptr<moon::transformational::Camera>>& getCameras()       const { return cameras; }
    const std::unordered_map<std::string, std::shared_ptr<moon::transformational::Group>>&  getGroups()        const { return groups; }
    const std::unordered_map<std::string, std::vector<moon::entities::SpotLight*>>&         getGroupLights()   const { return groupSpotLights; }
    moon::entities::DirectionalLight*                                                        getSunLight()      const { return sunLight.get(); }

    const std::vector<HierarchyNode>&                    getHierarchyRoots() const { return hierarchyRoots; }
    const std::unordered_map<std::string, EditorGroup>&  getEditorGroups()   const { return editorGroups; }

    const std::unordered_map<std::string, std::shared_ptr<moon::interfaces::Model>>& getModels() const { return models; }

    // Called from GUI panels
    void onSpawnPointLight()      { editorState.pendingSpawnType = EditorState::SpawnType::PointLight; }
    void onSpawnObject(const std::string& m) { editorState.pendingSpawnType = EditorState::SpawnType::Object; editorState.pendingSpawnModel = m; }
    void onSpawnObjectFromFile(const std::filesystem::path& p, float s, bool a) { spawnObjectFromFile(p, s, a); }
    const std::filesystem::path& getExternalPath() const { return ExternalPath; }
    uint32_t getResourceCount() const;
    void onSpawnSpotLight()       { editorState.pendingSpawnType = EditorState::SpawnType::SpotLight; }
    void onSpawnDirectionalLight(){ editorState.pendingSpawnType = EditorState::SpawnType::DirectionalLight; }
    void onSpawnIsotropicLight()  { editorState.pendingSpawnType = EditorState::SpawnType::IsotropicLight; }
    void onSpawnCamera()          { editorState.pendingSpawnType = EditorState::SpawnType::Camera; }
    void onSwitchToCamera(const std::string& n) { switchToCamera(n); }
    const std::string& getActiveCameraName() const { return activeCameraName; }
    bool isCameraGroup(const std::string& name) const { return cameras.count(name) > 0; }
    bool isLightGroup(const std::string& name) const;
    bool isNodeLocked(const std::string& name) const { return editorState.lockedNodes.count(name) > 0; }
    void onToggleLock(const std::string& name);
    void setLightEnabled(const std::string& groupName, bool enabled);
    void onFocusSelected()    { focusOnSelected(); }
    void onSelectObject(moon::transformational::Object* obj, const std::string& name) { selectObject(obj, name); }
    void onGroupSelected()    { groupSelected(); }
    void onDisbandGroup(const std::string& grpName) { disbandGroup(grpName); }
    void onToggleSelection(moon::transformational::Transformational* item,
                           const std::string& name, bool ctrlHeld)
        { toggleSelection(item, name, ctrlHeld); }
    void onClearSelection()   { clearSelection(); }
    void onSyncOutlines()     { syncOutlines(); requestUpdate(); }
    void onDeleteNode(const std::string& name);
    void onRenameNode(const std::string& oldName, const std::string& newName);
    void onReorderNode(const std::string& srcName, const std::string& dstName,
                       const std::string& parentGroup, bool insertAfter);
    void requestUpdate();
    float& animSpeed() { return animationSpeed; }
    void saveSceneToJson(const std::filesystem::path& jsonPath) const;

    moon::graphicsManager::GraphicsManager& getApp() const { return app; }
    moon::tests::GlfwWindow& getWindow() const { return window; }

    std::shared_ptr<moon::deferredGraphics::DeferredGraphics> baseGraphicsShared() const {
        auto it = graphics.find("base");
        return it != graphics.end() ? it->second : nullptr;
    }
};

#endif // SCENE_EDITOR_H
