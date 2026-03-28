#include "sceneEditor.h"
#include "editorGui.h"

#include <deferredGraphics/deferredGraphics.h>
#include <graphicsManager/graphicsManager.h>
#include <imguiGraphics/imguiGraphics.h>

#include <math/linearAlgebra.h>

#include <models/gltfmodel.h>
#include <models/plymodel.h>

#include <transformationals/cameras.h>
#include <transformationals/lights.h>
#include <transformationals/objects.h>
#include <transformationals/group.h>
#include <transformationals/transformational.h>

#include <entities/baseObject.h>

#include <utils/cursor.h>

#include <interfaces/light.h>
#include <interfaces/object.h>

#include <entities/baseCamera.h>
#include <entities/baseObject.h>
#include <entities/skyboxObject.h>
#include <entities/spotLight.h>
#include <entities/pointLight.h>
#include <entities/directionalLight.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <execution>
#include <random>
#include <fstream>

#include "json.hpp"
using json = nlohmann::json;

// Transformational* doesn't expose translation()/globalTransformation() getters —
// those are defined only in the concrete subclasses via DEFAULT_TRANSFORMATIONAL_GETTERS.
// We dispatch via dynamic_cast to get read access.

template<typename T>
static moon::math::vec3 worldPosImpl(T* t) {
    const auto& A = t->globalTransformation();
    const auto  c = t->translation().xyz(); // std::array<float,3>
    return {
        A[0][0]*c[0] + A[0][1]*c[1] + A[0][2]*c[2] + A[0][3],
        A[1][0]*c[0] + A[1][1]*c[1] + A[1][2]*c[2] + A[1][3],
        A[2][0]*c[0] + A[2][1]*c[1] + A[2][2]*c[2] + A[2][3]
    };
}

static moon::math::vec3 worldPos(moon::transformational::Transformational* t) {
    using namespace moon::transformational;
    if (auto* p = dynamic_cast<Object*>(t))  return worldPosImpl(p);
    if (auto* p = dynamic_cast<Group*>(t))   return worldPosImpl(p);
    if (auto* p = dynamic_cast<Camera*>(t))  return worldPosImpl(p);
    if (auto* p = dynamic_cast<Light*>(t))   return worldPosImpl(p);
    return {};
}

template<typename T>
static moon::math::quat worldRotImpl(T* t) {
    const auto& A = t->globalTransformation();
    float len0 = std::sqrt(A[0][0]*A[0][0] + A[1][0]*A[1][0] + A[2][0]*A[2][0]);
    float len1 = std::sqrt(A[0][1]*A[0][1] + A[1][1]*A[1][1] + A[2][1]*A[2][1]);
    float len2 = std::sqrt(A[0][2]*A[0][2] + A[1][2]*A[1][2] + A[2][2]*A[2][2]);
    if (len0 < 1e-6f || len1 < 1e-6f || len2 < 1e-6f) return t->rotation();
    moon::math::mat3 R;
    R[0][0] = A[0][0]/len0; R[0][1] = A[0][1]/len1; R[0][2] = A[0][2]/len2;
    R[1][0] = A[1][0]/len0; R[1][1] = A[1][1]/len1; R[1][2] = A[1][2]/len2;
    R[2][0] = A[2][0]/len0; R[2][1] = A[2][1]/len1; R[2][2] = A[2][2]/len2;
    return moon::math::convert(R) * t->rotation();
}

static moon::math::quat worldRot(moon::transformational::Transformational* t) {
    using namespace moon::transformational;
    if (auto* p = dynamic_cast<Object*>(t))  return worldRotImpl(p);
    if (auto* p = dynamic_cast<Group*>(t))   return worldRotImpl(p);
    if (auto* p = dynamic_cast<Camera*>(t))  return worldRotImpl(p);
    if (auto* p = dynamic_cast<Light*>(t))   return worldRotImpl(p);
    return moon::math::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

// Bake world translation AND rotation into a Transformational's own local state,
// then reset globalTransformation to identity.
template<typename T>
static void bakeWorldTransImpl(T* t, const moon::math::vec3& pos) {
    // Extract rotation from the upper-left 3x3 of globalTransformation.
    // Columns of that 3x3 are scaled basis vectors; normalizing them gives
    // the pure rotation matrix accumulated from all parent groups.
    const auto& A = t->globalTransformation();
    float len0 = std::sqrt(A[0][0]*A[0][0] + A[1][0]*A[1][0] + A[2][0]*A[2][0]);
    float len1 = std::sqrt(A[0][1]*A[0][1] + A[1][1]*A[1][1] + A[2][1]*A[2][1]);
    float len2 = std::sqrt(A[0][2]*A[0][2] + A[1][2]*A[1][2] + A[2][2]*A[2][2]);
    if (len0 > 1e-6f && len1 > 1e-6f && len2 > 1e-6f) {
        moon::math::mat3 R;
        R[0][0] = A[0][0]/len0; R[0][1] = A[0][1]/len1; R[0][2] = A[0][2]/len2;
        R[1][0] = A[1][0]/len0; R[1][1] = A[1][1]/len1; R[1][2] = A[1][2]/len2;
        R[2][0] = A[2][0]/len0; R[2][1] = A[2][1]/len1; R[2][2] = A[2][2]/len2;
        // world_rotation = group_accumulated_rotation * child_local_rotation
        t->rotation() = moon::math::convert(R) * t->rotation();
    }
    t->translation() = moon::math::quat(0.0f, pos[0], pos[1], pos[2]);
    t->setGlobalTransform(moon::math::mat4::identity()); // calls update()
}

static void bakeWorldTrans(moon::transformational::Transformational* t,
                           const moon::math::vec3& pos) {
    using namespace moon::transformational;
    if (auto* p = dynamic_cast<Object*>(t))  { bakeWorldTransImpl(p, pos); return; }
    if (auto* p = dynamic_cast<Group*>(t))   { bakeWorldTransImpl(p, pos); return; }
    if (auto* p = dynamic_cast<Camera*>(t))  { bakeWorldTransImpl(p, pos); return; }
    if (auto* p = dynamic_cast<Light*>(t))   { bakeWorldTransImpl(p, pos); return; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: spawn placement + marker cube factory
// ─────────────────────────────────────────────────────────────────────────────

static moon::math::vec3 cameraSpawnPos(
    const std::unordered_map<std::string, std::shared_ptr<moon::transformational::Camera>>& cameras,
    const std::string& activeName,
    float forwardOffset = 0.0f)
{
    auto it = cameras.find(activeName);
    if (it == cameras.end()) return {};
    auto pos = worldPos(it->second.get());
    if (forwardOffset != 0.0f) {
        auto q = worldRot(it->second.get());
        // Camera looks along -Z in local space; rotate by world quaternion
        moon::math::vec3 localFwd = {0.0f, 0.0f, -1.0f};
        // q * v * q^-1  (quaternion rotation of vector)
        moon::math::quat vq(0.0f, localFwd[0], localFwd[1], localFwd[2]);
        auto rotated = q * vq * moon::math::conjugate(q);
        moon::math::vec3 fwd = rotated.im();
        pos = pos + fwd * forwardOffset;
    }
    return pos;
}

static std::shared_ptr<moon::entities::BaseObject> makeMarkerCube(
    moon::interfaces::Model* cubeModel,
    const moon::math::vec4& color, float size = 0.08f)
{
    auto marker = std::make_shared<moon::entities::BaseObject>(cubeModel);
    marker->scale({ size, size, size });
    marker->setColor(color);
    marker->setBloom(color * 0.5f);
    return marker;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / destructor
// ─────────────────────────────────────────────────────────────────────────────

sceneEditor::sceneEditor(moon::graphicsManager::GraphicsManager& app,
                         moon::tests::GlfwWindow& window,
                         const std::filesystem::path& ExternalPath)
    : ExternalPath(ExternalPath),
      window(window),
      app(app),
      mouse(window),
      board(std::make_shared<controller>(window, glfwGetKey))
{
    mouse.control->sensitivity = 0.01f;
    board->sensitivity = 10.0f;
    create();
}

sceneEditor::~sceneEditor() {
    ImGui_ImplGlfw_Shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────

uint32_t sceneEditor::getResourceCount() const { return app.getResourceCount(); }

moon::deferredGraphics::DeferredGraphics* sceneEditor::baseGraphics() const {
    auto it = graphics.find("base");
    return it != graphics.end() ? it->second.get() : nullptr;
}

moon::transformational::Camera* sceneEditor::activeCamera() const {
    auto it = cameras.find(activeCameraName);
    return it != cameras.end() ? it->second.get() : nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create – full pipeline initialisation
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::create() {
    // ── Graphics pipeline initialisation ─────────────────────────────────────
    moon::deferredGraphics::Parameters params;
#ifndef MOON_SHADER_EMBED
    params.shadersPath          = std::filesystem::path(MOON_BUILD_PATH) / "spv/deferredGraphics";
    params.workflowsShadersPath = std::filesystem::path(MOON_BUILD_PATH) / "spv/workflows";
#endif
    params.extent               = window.sizes();
    params.layersCount          = 2;
    params.disableFaceCulling   = true;
    params.workflowsToAllocate  = {
        moon::deferredGraphics::Names::Bloom::name,
        moon::deferredGraphics::Names::Blur::name,
        moon::deferredGraphics::Names::Skybox::name,
        moon::deferredGraphics::Names::SSLR::name,
        moon::deferredGraphics::Names::SSAO::name,
        moon::deferredGraphics::Names::Shadow::name,
        moon::deferredGraphics::Names::Scattering::name,
        moon::deferredGraphics::Names::BoundingBox::name,
        moon::deferredGraphics::Names::Selector::name
    };

    graphics["base"] = std::make_shared<moon::deferredGraphics::DeferredGraphics>(params);
    app.setGraphics(graphics["base"].get());

    // ── Temporary camera for initial reset (replaced by JSON camera below) ──
    cameras["base"] = std::make_shared<moon::entities::BaseCamera>(45.0f, window.aspectRatio(), 0.1f);
    graphics["base"]->bind(cameras["base"]->camera());
    graphics["base"]->bind(mouse);
    graphics["base"]
        ->setEnable(moon::deferredGraphics::Names::Skybox::param,     true)
        .setEnable(moon::deferredGraphics::Names::Blur::param,        true)
        .setEnable(moon::deferredGraphics::Names::Bloom::param,       true)
        .setEnable(moon::deferredGraphics::Names::SSAO::param,        true)
        .setEnable(moon::deferredGraphics::Names::SSLR::param,        true)
        .setEnable(moon::deferredGraphics::Names::Scattering::param,  true)
        .setEnable(moon::deferredGraphics::Names::Shadow::param,      true)
        .setEnable(moon::deferredGraphics::Names::Selector::param,    true);
    graphics["base"]->reset();
    graphics["base"]->scatteringWorkflowParams().density = 0.05f;

    gui = std::make_shared<moon::imguiGraphics::ImguiGraphics>(app.getInstance(), app.getImageCount());
    app.setGraphics(gui.get());
    if (ImGui_ImplGlfw_InitForVulkan(window, true)) {
        gui->reset();
    }

    // ── Load scene from JSON (creates camera, objects, lights) ───────────────
    auto scenePath = ExternalPath / "tests/sceneEditor/defaultScene.json";
    auto u8 = scenePath.u8string();
    std::snprintf(editorState.sceneFilePath, sizeof(editorState.sceneFilePath),
                  "%s", std::string(u8.begin(), u8.end()).c_str());
    loadSceneFromJson(scenePath);

    // Re-bind camera (loadSceneFromJson may have replaced it)
    graphics["base"]->bind(cameras["base"]->camera());

    // ── Camera group (red pyramid marker + camera as children) ───────────────
    {
        auto marker = makeMarkerCube(models["pyramid"].get(),
            moon::math::vec4{ 1.0f, 0.2f, 0.2f, 1.0f }, 0.1f);

        auto grp = std::make_shared<moon::transformational::Group>();
        grp->setGlobalTransform(moon::math::mat4::identity());
        if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(cameras["base"].get())) {
            grp->translation() = cam->translation();
            // Group rotation stays identity — look rotation stays on camera
            // so that rotateY yaw axis (0,0,1) stays aligned with world "up"
            grp->add(marker.get());
            grp->add(cam);
            cam->translation() = moon::math::quat(0.0f, 0.0f, 0.0f, 0.0f);
            grp->update();
        }
        groups["base"] = grp;

        graphics["base"]->bind(marker->object());
        markerOwners[marker->object()] = { grp.get(), "base" };

        EditorGroup eg;
        eg.group = grp.get();
        eg.pivotCube = std::move(marker);
        editorGroups["base"] = std::move(eg);

        for (auto it = hierarchyRoots.begin(); it != hierarchyRoots.end(); ++it) {
            if (it->name == "base" && it->kind == NodeKind::Camera) {
                it->kind = NodeKind::Group;
                it->ptr  = grp.get();
                break;
            }
        }
        nodeNames[grp.get()] = "base";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Models
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::createModels() {
    const auto rc          = app.getResourceCount();
    const auto modelPath   = ExternalPath / "dependences/model";
    const auto glTFSamples = modelPath / "glTF-Sample-Models/2.0";

    models["bee"]       = std::make_shared<moon::models::GltfModel>(modelPath / "glb/Bee.glb",       2 * rc);
    models["butterfly"] = std::make_shared<moon::models::GltfModel>(modelPath / "glb/butterfly.glb", rc);
    models["robot"]     = std::make_shared<moon::models::GltfModel>(modelPath / "glb/Robot.glb",     rc);
    models["ufo"]       = std::make_shared<moon::models::GltfModel>(modelPath / "glb/RetroUFO.glb");

    models["sponza"]    = std::make_shared<moon::models::GltfModel>(
        glTFSamples / "Sponza/glTF/Sponza.gltf");
    models["helmet"]    = std::make_shared<moon::models::GltfModel>(
        glTFSamples / "DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    models["cube"]      = std::make_shared<moon::models::PlyModel>(modelPath / "ply/cube.ply");
    models["pyramid"]   = std::make_shared<moon::models::PlyModel>(modelPath / "ply/pyramid.ply");

    for (auto& [_, m] : models)
        graphics["base"]->create(m.get());
}

// ─────────────────────────────────────────────────────────────────────────────
// Objects
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::registerNode(NodeKind kind, const std::string& name,
                               moon::transformational::Transformational* ptr) {
    hierarchyRoots.push_back({ kind, name, ptr });
    nodeNames[ptr] = name;
}

void sceneEditor::groupSelected() {
    auto& sel = editorState.selection;
    if (sel.size() < 2) return;

    auto isGroupable = [this](const HierarchyNode& n) -> bool {
        if (editorState.lockedNodes.count(n.name)) return false;
        if (n.kind == NodeKind::SkyboxObject) return false;
        return true;
    };

    // Compute center of mass of all groupable selected top-level nodes
    moon::math::vec3 center{ 0.0f, 0.0f, 0.0f };
    int count = 0;
    for (const auto& n : hierarchyRoots) {
        if (sel.count(n.ptr) && isGroupable(n)) {
            auto wp = worldPos(n.ptr);
            center[0] += wp[0]; center[1] += wp[1]; center[2] += wp[2];
            ++count;
        }
    }
    if (count < 2) return;
    center[0] /= count; center[1] /= count; center[2] /= count;

    std::string grpName = "Group_" + std::to_string(++groupNameCounter);
    auto group = std::make_shared<moon::transformational::Group>();
    group->setGlobalTransform(moon::math::mat4::identity());
    groups[grpName] = group;

    // Place group origin at center of mass so children keep their world positions
    group->translate(center);

    EditorGroup eg;
    eg.group = group.get();

    // Create yellow pivot cube at group origin (local 0,0,0 = world center)
    auto pivot = std::make_shared<moon::entities::BaseObject>(models["cube"].get());
    pivot->scale({ 0.05f, 0.05f, 0.05f });
    pivot->setColor(moon::math::vec4{ 1.0f, 0.85f, 0.0f, 1.0f });
    pivot->setBloom(moon::math::vec4{ 0.6f, 0.4f, 0.0f, 1.0f });
    group->add(pivot.get());
    baseGraphics()->bind(pivot->object());
    markerOwners[pivot->object()] = { group.get(), grpName };
    eg.pivotCube = std::move(pivot);

    // Bake each child's world position into its local translation relative to group center
    for (auto it = hierarchyRoots.begin(); it != hierarchyRoots.end(); ) {
        if (sel.count(it->ptr) && isGroupable(*it)) {
            auto wp = worldPos(it->ptr); // compute before modifying
            // Set local offset = worldPos - center; after group.add() child renders at worldPos
            bakeWorldTrans(it->ptr, { wp[0]-center[0], wp[1]-center[1], wp[2]-center[2] });
            group->add(it->ptr); // sets child.globalTransform = translate(center) → renders at wp
            eg.children.push_back(*it);
            it = hierarchyRoots.erase(it);
        } else {
            ++it;
        }
    }

    editorGroups[grpName] = std::move(eg);
    registerNode(NodeKind::Group, grpName, group.get());

    clearSelection();
    requestUpdate();
}

void sceneEditor::disbandGroup(const std::string& grpName) {
    auto egIt = editorGroups.find(grpName);
    if (egIt == editorGroups.end()) return;

    // Atomic light/camera groups cannot be disbanded
    if (isLightGroup(grpName) || isCameraGroup(grpName)) return;

    // Clear selection first: sel may hold the Group* or its children's ptrs.
    // dynamic_cast in syncOutlines() would dereference them after the Group is freed.
    clearSelection();

    auto* grp = egIt->second.group;

    // Bake each child's world position before removing from group,
    // so re-grouping later won't reset them to their old local origin.
    for (auto& child : egIt->second.children) {
        auto wp = worldPos(child.ptr); // while globalTransform still = group.modelMatrix
        grp->remove(child.ptr);
        bakeWorldTrans(child.ptr, wp);
        hierarchyRoots.push_back(child);
    }

    // Remove pivot cube from the engine group and graphics pipeline.
    // Must wait for GPU to finish before destroying buffers still in use.
    if (egIt->second.pivotCube) {
        app.deviceWaitIdle();
        markerOwners.erase(egIt->second.pivotCube->object());
        grp->remove(egIt->second.pivotCube.get());
        baseGraphics()->remove(egIt->second.pivotCube->object());
        // shared_ptr goes out of scope with editorGroups.erase() below
    }

    // Remove the group node from hierarchyRoots
    hierarchyRoots.erase(
        std::remove_if(hierarchyRoots.begin(), hierarchyRoots.end(),
            [&](const HierarchyNode& n) {
                return n.kind == NodeKind::Group && n.name == grpName;
            }),
        hierarchyRoots.end());

    nodeNames.erase(egIt->second.group);
    editorGroups.erase(egIt);
    groups.erase(grpName);

    requestUpdate();
}

void sceneEditor::toggleSelection(moon::transformational::Transformational* item,
                                  const std::string& name, bool ctrlHeld) {
    auto& sel = editorState.selection;

    if (!ctrlHeld) {
        // Single-select: replace selection
        syncOutlines(); // clear old outlines first via syncOutlines with cleared set
        sel.clear();
        sel.insert(item);
        editorState.selectedObject = dynamic_cast<moon::transformational::Object*>(item);
        editorState.selectedName   = name;
    } else {
        // Ctrl: toggle membership
        if (sel.count(item)) {
            sel.erase(item);
            if (editorState.selectedObject == dynamic_cast<moon::transformational::Object*>(item)) {
                editorState.selectedObject = nullptr;
                editorState.selectedName.clear();
                // Try to find another Object in the remaining selection as primary
                for (auto* s : sel) {
                    if (auto* o = dynamic_cast<moon::transformational::Object*>(s)) {
                        editorState.selectedObject = o;
                        editorState.selectedName   = nodeNames.count(s) ? nodeNames[s] : "";
                        break;
                    }
                }
            }
        } else {
            sel.insert(item);
            if (!editorState.selectedObject) {
                editorState.selectedObject = dynamic_cast<moon::transformational::Object*>(item);
                editorState.selectedName   = name;
            }
        }
    }

    syncOutlines();
    requestUpdate();
}

void sceneEditor::clearSelection() {
    editorState.selection.clear();
    editorState.selectedObject = nullptr;
    editorState.selectedName.clear();
    syncOutlines();
}

// Collect all Object* descendants of an EditorGroup (recursive for nested groups)
static void collectGroupObjects(
    const std::unordered_map<std::string, sceneEditor::EditorGroup>& editorGroups,
    const sceneEditor::EditorGroup& eg,
    std::unordered_set<moon::transformational::Object*>& out)
{
    for (const auto& child : eg.children) {
        if (child.kind == NodeKind::Group) {
            // Recurse into nested groups
            auto* grp = dynamic_cast<moon::transformational::Group*>(child.ptr);
            for (const auto& [_cname, childEg] : editorGroups) {
                if (childEg.group == grp) {
                    collectGroupObjects(editorGroups, childEg, out);
                    break;
                }
            }
        } else {
            if (auto* obj = dynamic_cast<moon::transformational::Object*>(child.ptr))
                out.insert(obj);
        }
    }
}

void sceneEditor::syncOutlines() {
    const auto& sel  = editorState.selection;
    const auto& col  = editorState.outlineColor;
    const bool  en   = editorState.outlineEnabled;
    const moon::math::vec4 secCol{ 0.4f, 0.8f, 0.4f, 1.0f }; // secondary selection colour

    // Build the set of objects to outline
    std::unordered_set<moon::transformational::Object*> outlined;   // all to outline
    std::unordered_set<moon::transformational::Object*> primary;    // primary colour

    for (auto* ptr : sel) {
        if (auto* obj = dynamic_cast<moon::transformational::Object*>(ptr)) {
            outlined.insert(obj);
            if (editorState.selectedObject == obj) primary.insert(obj);
        } else if (auto* grp = dynamic_cast<moon::transformational::Group*>(ptr)) {
            // Outline all descendants of selected groups with secondary colour
            for (const auto& [_ename, eg] : editorGroups) {
                if (eg.group == grp) {
                    collectGroupObjects(editorGroups, eg, outlined);
                    break;
                }
            }
        }
    }

    auto applyObj = [&](const std::shared_ptr<moon::transformational::Object>& obj) {
        auto* bo = dynamic_cast<moon::entities::BaseObject*>(obj.get());
        if (!bo) return;
        if (outlined.count(obj.get())) {
            bool isPrimary = primary.count(obj.get()) > 0;
            bo->setOutlining(en, 0.03f, isPrimary ? col : secCol);
        } else {
            bo->setOutlining(false);
        }
    };

    for (auto& [_, obj] : objects) applyObj(obj);
}

void sceneEditor::createObjects() {
    const auto rc = app.getResourceCount();

    // ── Backdrop (locked by default) ────────────────────────────────────────
    objects["sponza"] = std::make_shared<moon::entities::BaseObject>(models["sponza"].get());
    objects["sponza"]->rotate(moon::math::radians(90.0f), {1.0f,0.0f,0.0f}).scale({3.0f,3.0f,3.0f});

    // ── Animated objects ─────────────────────────────────────────────────────
    objects["bee"] = std::make_shared<moon::entities::BaseObject>(models["bee"].get(), 0, rc);
    objects["bee"]->translate({5.0f,0.0f,0.0f})
                  .rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f})
                  .scale({0.2f,0.2f,0.2f});
    static_cast<moon::entities::BaseObject*>(objects["bee"].get())->animation.play(0);

    objects["butterfly"] = std::make_shared<moon::entities::BaseObject>(models["butterfly"].get(), 0, rc);
    objects["butterfly"]->rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f}).scale({20.0f,20.0f,20.0f});
    static_cast<moon::entities::BaseObject*>(objects["butterfly"].get())->animation.play(0);

    objects["robot"] = std::make_shared<moon::entities::BaseObject>(models["robot"].get(), 0, rc);
    objects["robot"]->scale(25.0f)
                    .rotate(moon::math::quat(0.5f,0.5f,-0.5f,-0.5f))
                    .rotate(moon::math::radians(180.0f),{0.0f,0.0f,1.0f})
                    .translate(moon::math::vec3(-30.0f,11.0f,10.0f));
    static_cast<moon::entities::BaseObject*>(objects["robot"].get())->animation.play(1);

    // ── Helmet prop (static, interesting model) ───────────────────────────────
    objects["helmet"] = std::make_shared<moon::entities::BaseObject>(models["helmet"].get());
    objects["helmet"]->scale(1.0f).rotate(moon::math::quat(0.5f,0.5f,-0.5f,-0.5f))
                     .translate(moon::math::vec3(0.0f,3.0f,12.0f));

    // ── UFO with spotlight group ──────────────────────────────────────────────
    objects["ufo"] = std::make_shared<moon::entities::BaseObject>(models["ufo"].get());
    objects["ufo"]->rotate(moon::math::radians(90.0f),{1.0f,0.0f,0.0f});
    groups["ufo"] = std::make_shared<moon::transformational::Group>();
    groups["ufo"]->setGlobalTransform(moon::math::mat4::identity());
    groups["ufo"]->add(objects["ufo"].get());
    groups["ufo"]->translate({0.0f,0.0f,8.0f});

    // ── Skybox ────────────────────────────────────────────────────────────────
    skyboxObjects["stars"] = std::make_shared<moon::entities::SkyboxObject>(
        moon::utils::vkDefault::Paths{
            ExternalPath / "dependences/texture/skybox1/left.png",
            ExternalPath / "dependences/texture/skybox1/right.png",
            ExternalPath / "dependences/texture/skybox1/front.png",
            ExternalPath / "dependences/texture/skybox1/back.png",
            ExternalPath / "dependences/texture/skybox1/top.png",
            ExternalPath / "dependences/texture/skybox1/bottom.png"
        });
    skyboxObjects["stars"]->scale({200.0f,200.0f,200.0f});

    // ── Bind everything to renderer ───────────────────────────────────────────
    for (auto& [_, graph] : graphics) {
        for (auto& [_, obj] : objects)       graph->bind(obj->object());
        for (auto& [_, obj] : skyboxObjects) graph->bind(obj->object());
    }

    // ── Register hierarchy nodes ──────────────────────────────────────────────
    registerNode(NodeKind::Camera,       "base",      cameras["base"].get());
    registerNode(NodeKind::Object,       "bee",       objects["bee"].get());
    registerNode(NodeKind::Object,       "butterfly", objects["butterfly"].get());
    registerNode(NodeKind::Object,       "robot",     objects["robot"].get());
    registerNode(NodeKind::Object,       "helmet",    objects["helmet"].get());
    registerNode(NodeKind::Object,       "ufo",       objects["ufo"].get());
    registerNode(NodeKind::Object,       "sponza",    objects["sponza"].get());
    registerNode(NodeKind::SkyboxObject, "stars",     skyboxObjects["stars"].get());

    // ── Default locked nodes ──────────────────────────────────────────────────
    editorState.lockedNodes.insert("sponza");
    editorState.lockedNodes.insert("stars");
    editorState.lockedNodes.insert("sun");
}

// ─────────────────────────────────────────────────────────────────────────────
// Lights
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::createLights() {
    using namespace moon::entities;

    // ── UFO Spotlight (atomic light unit inside ufo group) ────────────────────
    auto ufoLight = std::make_shared<SpotLight>(
        moon::math::vec4(1.0f,0.85f,0.6f,1.0f),
        SpotLight::Props{ true, true, 0.2f, 10.0f, 0.3f });
    lightSources.push_back(ufoLight);
    groupSpotLights["ufo"].push_back(static_cast<SpotLight*>(ufoLight.get()));

    auto ufoLightMarker = makeMarkerCube(models["pyramid"].get(),
        moon::math::vec4{1.0f, 0.85f, 0.6f, 1.0f}, 0.08f);
    auto ufoLightGroup = std::make_shared<moon::transformational::Group>();
    ufoLightGroup->setGlobalTransform(moon::math::mat4::identity());
    ufoLightGroup->add(ufoLight.get());
    ufoLightGroup->add(ufoLightMarker.get());
    groups["ufoLight"] = ufoLightGroup;
    groups["ufo"]->add(ufoLightGroup.get());

    {
        EditorGroup eg;
        eg.group = ufoLightGroup.get();
        eg.pivotCube = ufoLightMarker;
        eg.lightEntity = ufoLight.get();
        editorGroups["ufoLight"] = std::move(eg);
        nodeNames[ufoLightGroup.get()] = "ufoLight";
    }

    // ── Directional sun (atomic light unit) ──────────────────────────────────
    sunLight = std::make_shared<DirectionalLight>(
        moon::math::vec4(1.0f, 0.95f, 0.85f, 1.0f),
        DirectionalLight::Props{ true, 3.0f, 0.0f, 300.0f, 300.0f, 500.0f });

    auto sunMarker = makeMarkerCube(models["pyramid"].get(),
        moon::math::vec4{1.0f, 0.85f, 0.1f, 1.0f}, 0.15f);

    groups["sun"] = std::make_shared<moon::transformational::Group>();
    groups["sun"]->setGlobalTransform(moon::math::mat4::identity());
    groups["sun"]->add(sunLight.get());
    groups["sun"]->add(sunMarker.get());
    groups["sun"]->translate(moon::math::vec3(0.0f, 0.0f, 100.0f));

    {
        EditorGroup eg;
        eg.group = groups["sun"].get();
        eg.pivotCube = sunMarker;
        eg.lightEntity = sunLight.get();
        editorGroups["sun"] = std::move(eg);
        nodeNames[groups["sun"].get()] = "sun";
    }

    // ── Bind to renderer ──────────────────────────────────────────────────────
    for (auto& [_, graph] : graphics) {
        for (auto& source : lightSources) graph->bind(source->light());
        graph->bind(sunLight->light());
        graph->bind(ufoLightMarker->object());
        graph->bind(sunMarker->object());
    }

    // ── Marker owners for picking ─────────────────────────────────────────────
    markerOwners[ufoLightMarker->object()] = { ufoLightGroup.get(), "ufoLight" };
    markerOwners[sunMarker->object()] = { groups["sun"].get(), "sun" };

    // ── Register sun as top-level group node ──────────────────────────────────
    registerNode(NodeKind::Group, "sun", groups["sun"].get());
    // ufoLight is NOT top-level — it lives inside the ufo group
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::resize() {
    if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera())) {
        cam->setProjMatrix(moon::math::perspective(
            moon::math::radians(45.0f), window.aspectRatio(), 0.1f));
    }
    auto winSz = window.sizes();
    float s = editorState.resolutionScale;
    graphics["base"]->parameters().extent = moon::math::vec2u(
        std::max(1u, static_cast<uint32_t>(winSz[0] * s)),
        std::max(1u, static_cast<uint32_t>(winSz[1] * s))
    );
    for (auto& [_, g] : graphics) g->reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// Update frame
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::updateFrame(uint32_t frameNumber, float inFrameTime) {
    frameTime = inFrameTime;

    // Process deferred spawns (must happen outside render pass)
    if (editorState.pendingSpawn.pending) {
        auto& ps = editorState.pendingSpawn;
        app.deviceWaitIdle();
        try {
            spawnObjectFromFile(ps.filePath, ps.scale, ps.animated);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Failed to load model: %s\n", e.what());
        }
        ps.pending = false;
    }
    if (editorState.pendingSpawnType != EditorState::SpawnType::None) {
        app.deviceWaitIdle();
        switch (editorState.pendingSpawnType) {
            case EditorState::SpawnType::PointLight:       spawnPointLight(); break;
            case EditorState::SpawnType::SpotLight:        spawnSpotLight(); break;
            case EditorState::SpawnType::DirectionalLight: spawnDirectionalLight(); break;
            case EditorState::SpawnType::IsotropicLight:   spawnIsotropicLight(); break;
            case EditorState::SpawnType::Camera:           spawnCamera(); break;
            case EditorState::SpawnType::Object:           spawnObject(editorState.pendingSpawnModel); break;
            default: break;
        }
        editorState.pendingSpawnType = EditorState::SpawnType::None;
    }

    // Process deferred resolution change
    if (editorState.pendingResolutionChange) {
        editorState.pendingResolutionChange = false;
        app.deviceWaitIdle();
        auto winSz = window.sizes();
        float s = editorState.resolutionScale;
        graphics["base"]->parameters().extent = moon::math::vec2u(
            std::max(1u, static_cast<uint32_t>(winSz[0] * s)),
            std::max(1u, static_cast<uint32_t>(winSz[1] * s))
        );
        for (auto& [_, g] : graphics) g->reset();
    }

    glfwPollEvents();

    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    makeGui();

    if (!io.WantCaptureMouse)    mouseEvent();
    if (!io.WantCaptureKeyboard) keyboardEvent();

    // Sync marker cube to camera's local position within the group
    // (both are children of the same marker group).
    {
        auto egIt = editorGroups.find(activeCameraName);
        if (egIt != editorGroups.end()) {
            if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera())) {
                if (egIt->second.pivotCube) {
                    egIt->second.pivotCube->translation() = cam->translation();
                    egIt->second.pivotCube->rotation() = cam->rotation();
                }
                egIt->second.group->update();
            }
        }
    }

    // Animate glTF skeletal/morph animations (bee, butterfly, robot)
    float animDt = animationSpeed * frameTime;

    std::for_each(std::execution::par_unseq, objects.begin(), objects.end(),
        [frameNumber, animDt](auto& pair) {
            auto* obj = dynamic_cast<moon::entities::BaseObject*>(pair.second.get());
            if (obj) obj->updateAnimation(frameNumber, animDt);
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// GUI – delegate to editor::gui panels
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::makeGui() {
    editor::gui::drawAll(*this);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mouse events
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::mouseEvent() {
    const float sens = mouse.control->sensitivity;
    const auto& cursorBuffer = mouse.cursor->read();
    const uint32_t primitiveNumber = cursorBuffer.info.number;
    const auto xy = window.mousePose();

    // Camera rotation
    if (mouse.control->pressed(GLFW_MOUSE_BUTTON_LEFT) ||
        mouse.control->pressed(GLFW_MOUSE_BUTTON_RIGHT))
    {
        const auto delta = mouse.pose - xy;
        if (auto* cam = activeCamera()) {
            cam->rotateX(static_cast<float>(sens * delta[1]));
            cam->rotateY(static_cast<float>(sens * delta[0]));
        }
    }

    if (mouse.control->pressed(GLFW_MOUSE_BUTTON_LEFT)) {
        const auto sizes = window.sizes();
        mouse.cursor->update(static_cast<float>(xy[0] / sizes[0]),
                             static_cast<float>(xy[1] / sizes[1]));
    }
    mouse.pose = xy;

    // Object picking on release
    // Ctrl+click = instant toggle (add/remove from multi-selection)
    // Normal click = double-click to select (avoids deselection when rotating camera)
    if (mouse.control->released(GLFW_MOUSE_BUTTON_LEFT)) {
        bool ctrlHeld = board->pressed(GLFW_KEY_LEFT_CONTROL);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastClickTime).count();
        double dx = xy[0] - lastClickPos[0];
        double dy = xy[1] - lastClickPos[1];
        double dist = std::sqrt(dx * dx + dy * dy);
        bool isDoubleClick = (elapsed < doubleClickMaxSeconds && dist < doubleClickMaxDist);

        lastClickTime = now;
        lastClickPos = xy;

        bool doPick = ctrlHeld || isDoubleClick;
        if (doPick) {
            bool found = false;
            for (auto& [name, obj] : objects) {
                if (!obj->object()->comparePrimitive(primitiveNumber)) continue;
                toggleSelection(obj.get(), name, ctrlHeld);
                found = true;
                break;
            }
            // Marker cubes: pivot cubes, light markers, camera markers
            if (!found) {
                for (auto& [markerObj, ownerPair] : markerOwners) {
                    if (!markerObj->comparePrimitive(primitiveNumber)) continue;
                    // Camera group: click without Ctrl switches to that camera
                    if (!ctrlHeld && cameras.count(ownerPair.second)) {
                        switchToCamera(ownerPair.second);
                        found = true;
                        break;
                    }
                    toggleSelection(ownerPair.first, ownerPair.second, ctrlHeld);
                    found = true;
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Keyboard events
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::keyboardEvent() {
    const float sens = board->sensitivity * frameTime;

    if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera())) {
        const auto view = cam->getViewMatrix();
        if (!board->pressed(GLFW_KEY_LEFT_CONTROL)) {
            // Transform world-space view directions to group-local space
            // so WASD works correctly when camera is in a rotated group.
            const auto invGlobal = moon::math::inverse(cam->globalTransformation());
            auto toLocal = [&](const moon::math::vec3& worldDir) -> moon::math::vec3 {
                auto v = invGlobal * moon::math::vec4(worldDir[0], worldDir[1], worldDir[2], 0.0f);
                return v.dvec();
            };
            auto* activeCam = activeCamera();
            if (board->pressed(GLFW_KEY_A)) activeCam->translate(toLocal(-sens * view[0].dvec()));
            if (board->pressed(GLFW_KEY_D)) activeCam->translate(toLocal( sens * view[0].dvec()));
            if (board->pressed(GLFW_KEY_W)) activeCam->translate(toLocal(-sens * view[2].dvec()));
            if (board->pressed(GLFW_KEY_S)) activeCam->translate(toLocal( sens * view[2].dvec()));
            if (board->pressed(GLFW_KEY_E)) activeCam->translate(toLocal( sens * view[1].dvec()));
            if (board->pressed(GLFW_KEY_Q)) activeCam->translate(toLocal(-sens * view[1].dvec()));
        }
    }

    // Ctrl+S saves scene
    if (board->pressed(GLFW_KEY_LEFT_CONTROL) && board->released(GLFW_KEY_S))
        saveSceneToJson(std::filesystem::path(editorState.sceneFilePath));

    // Focus on selected object
    if (board->released(GLFW_KEY_F)) focusOnSelected();

    // Escape closes
    if (board->released(GLFW_KEY_ESCAPE)) window.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// Select an object (sets outline, updates editor state)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::selectObject(moon::transformational::Object* obj, const std::string& name) {
    if (editorState.selectedObject == obj && editorState.selection.size() == 1) return;
    toggleSelection(obj, name, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Focus camera on the selected object
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::focusOnSelected() {
    auto* cam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera());
    if (!cam) return;

    moon::math::vec3 target{0.0f};
    bool hasTarget = false;

    if (editorState.selectedObject) {
        target    = editorState.selectedObject->translation().im();
        hasTarget = true;
    } else {
        // Try to find a selected group
        for (const auto& [name, eg] : editorGroups) {
            if (editorState.selection.count(eg.group)) {
                target    = eg.group->translation().im();
                hasTarget = true;
                break;
            }
        }
    }
    if (!hasTarget) return;

    moon::math::vec3 offset = { 0.0f, 0.0f, 8.0f };
    // Convert world-space target to group-local space
    auto worldTarget = target + offset;
    const auto invGlobal = moon::math::inverse(cam->globalTransformation());
    auto local4 = invGlobal * moon::math::vec4(worldTarget[0], worldTarget[1], worldTarget[2], 1.0f);
    cam->translation() = moon::math::quat(0.0f, local4.dvec());
    cam->update();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: Object
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnObject(const std::string& modelName) {
    auto modelIt = models.find(modelName);
    if (modelIt == models.end()) return;

    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);
    std::string name = modelName + "_" + std::to_string(++objectNameCounter);

    auto obj = std::make_shared<moon::entities::BaseObject>(modelIt->second.get());
    obj->translate(pos);
    objects[name] = obj;
    objectModelKeys[name] = modelName;

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics)
        graph->bind(obj->object());

    registerNode(NodeKind::Object, name, obj.get());
    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: Object from arbitrary file path
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnObjectFromFile(const std::filesystem::path& filePath, float scale, bool animated) {
    // Derive a short name from the filename stem
    auto u8stem = filePath.stem().u8string();
    std::string stem(u8stem.begin(), u8stem.end());
    std::string modelKey = stem + "_dyn_" + std::to_string(++objectNameCounter);

    // Create the model
    auto u8ext = filePath.extension().u8string();
    std::string ext(u8ext.begin(), u8ext.end());
    std::shared_ptr<moon::interfaces::Model> model;
    if (ext == ".ply") {
        model = std::make_shared<moon::models::PlyModel>(filePath);
    } else if (ext == ".glb" || ext == ".gltf") {
        uint32_t instanceCount = animated ? app.getResourceCount() : 1;
        model = std::make_shared<moon::models::GltfModel>(filePath, instanceCount);
    } else {
        return;
    }

    // Upload to GPU — may fail for invalid models
    for (auto& [_, graph] : graphics)
        graph->create(model.get());

    // GltfModel sets vertexType to 'non' only if left at default;
    // constructor sets 'animated', loadFromFile corrects to pbr or animated.
    // If load failed, type may still be 'animated' but no geometry exists.
    // GltfModel::render guards against NULL vertex buffer, so it's safe
    // to add — the object simply won't be visible.

    models[modelKey] = model;

    // Track model path for serialization
    auto relPath = std::filesystem::relative(filePath, ExternalPath);
    auto u8rel = relPath.u8string();
    std::string relStr(u8rel.begin(), u8rel.end());
    std::replace(relStr.begin(), relStr.end(), '\\', '/');
    modelPaths[modelKey] = relStr;
    objectModelKeys[modelKey] = modelKey;

    // Normalize model size: fit into 1x1x1 box, then apply user scale
    auto bb = model->boundingBox();
    moon::math::vec3 extent = {
        bb.max[0] - bb.min[0],
        bb.max[1] - bb.min[1],
        bb.max[2] - bb.min[2]
    };
    float maxExtent = std::max({ extent[0], extent[1], extent[2] });
    float normalizeScale = (maxExtent > 0.0f) ? (1.0f / maxExtent) : 1.0f;
    float finalScale = normalizeScale * scale;

    // Create the object
    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);
    uint32_t objInstanceCount = animated ? app.getResourceCount() : 1;
    auto obj = std::make_shared<moon::entities::BaseObject>(model.get(), 0, objInstanceCount);
    obj->translate(pos);
    obj->scale({finalScale, finalScale, finalScale});
    objects[modelKey] = obj;

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics)
        graph->bind(obj->object());

    registerNode(NodeKind::Object, modelKey, obj.get());
    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: PointLight (marker cube + light in a group)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnPointLight() {
    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);

    std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.3f, 1.0f);
    moon::math::vec4 color{ dist(rng), dist(rng), dist(rng), 1.0f };

    auto light = std::make_shared<moon::entities::PointLight>(
        moon::entities::PointLight::Props{ color, 8.0f, 15.0f, 0.5f });

    std::string lightName = "Point_" + std::to_string(pointLightPtrs.size());
    pointLightPtrs.push_back(light.get());
    lightSources.push_back(light);

    auto marker = makeMarkerCube(models["cube"].get(), color);

    auto group = std::make_shared<moon::transformational::Group>();
    group->setGlobalTransform(moon::math::mat4::identity());
    group->add(light.get());
    group->add(marker.get());
    group->translate(pos);
    groups[lightName] = group;

    EditorGroup eg;
    eg.group = group.get();
    eg.pivotCube = marker;
    eg.lightEntity = light.get();
    editorGroups[lightName] = std::move(eg);

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics) {
        graph->bind(light->light());
        graph->bind(marker->object());
    }

    markerOwners[marker->object()] = { group.get(), lightName };
    registerNode(NodeKind::Group, lightName, group.get());
    nodeNames[group.get()] = lightName;

    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: SpotLight (pyramid marker + light in a group)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnSpotLight() {
    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);

    moon::math::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f };
    std::string lightName = "Spot_" + std::to_string(++spotLightNameCounter);

    auto light = std::make_shared<moon::entities::SpotLight>(
        moon::entities::SpotLight::Coloring(color));
    lightSources.push_back(light);

    auto marker = makeMarkerCube(models["pyramid"].get(),
        moon::math::vec4{ 1.0f, 1.0f, 0.5f, 1.0f });

    auto group = std::make_shared<moon::transformational::Group>();
    group->setGlobalTransform(moon::math::mat4::identity());
    group->add(light.get());
    group->add(marker.get());
    group->translate(pos);
    groups[lightName] = group;

    EditorGroup eg;
    eg.group = group.get();
    eg.pivotCube = marker;
    eg.lightEntity = light.get();
    editorGroups[lightName] = std::move(eg);

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics) {
        graph->bind(light->light());
        graph->bind(marker->object());
    }

    markerOwners[marker->object()] = { group.get(), lightName };
    registerNode(NodeKind::Group, lightName, group.get());
    nodeNames[group.get()] = lightName;

    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: DirectionalLight (pyramid marker + light in a group)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnDirectionalLight() {
    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);

    moon::math::vec4 color{ 1.0f, 0.95f, 0.8f, 1.0f };
    std::string lightName = "Dir_" + std::to_string(++dirLightNameCounter);

    auto light = std::make_shared<moon::entities::DirectionalLight>(color);
    lightSources.push_back(light);

    auto marker = makeMarkerCube(models["pyramid"].get(),
        moon::math::vec4{ 1.0f, 0.7f, 0.2f, 1.0f });

    auto group = std::make_shared<moon::transformational::Group>();
    group->setGlobalTransform(moon::math::mat4::identity());
    group->add(light.get());
    group->add(marker.get());
    group->translate(pos);
    groups[lightName] = group;

    EditorGroup eg;
    eg.group = group.get();
    eg.pivotCube = marker;
    eg.lightEntity = light.get();
    editorGroups[lightName] = std::move(eg);

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics) {
        graph->bind(light->light());
        graph->bind(marker->object());
    }

    markerOwners[marker->object()] = { group.get(), lightName };
    registerNode(NodeKind::Group, lightName, group.get());
    nodeNames[group.get()] = lightName;

    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: IsotropicLight (6 SpotLights in a Group, + marker cube)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnIsotropicLight() {
    auto pos = cameraSpawnPos(cameras, activeCameraName, 5.0f);

    moon::math::vec4 color{ 0.8f, 0.9f, 1.0f, 1.0f };
    std::string lightName = "Iso_" + std::to_string(++isoLightNameCounter);

    auto isoLight = std::make_shared<moon::entities::IsotropicLight>(
        moon::entities::IsotropicLight::Props{ color, 30.0f });
    isoLight->setGlobalTransform(moon::math::mat4::identity());

    auto marker = makeMarkerCube(models["cube"].get(),
        moon::math::vec4{ 0.5f, 0.8f, 1.0f, 1.0f });
    isoLight->add(marker.get());
    isoLight->translate(pos);

    // IsotropicLight IS a Group — store it as such
    groups[lightName] = isoLight;

    EditorGroup eg;
    eg.group = isoLight.get();
    eg.pivotCube = marker;
    eg.lightEntity = isoLight.get();
    editorGroups[lightName] = std::move(eg);

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics) {
        for (auto* l : isoLight->getLights())
            graph->bind(l);
        graph->bind(marker->object());
    }

    markerOwners[marker->object()] = { isoLight.get(), lightName };
    registerNode(NodeKind::Group, lightName, isoLight.get());
    nodeNames[isoLight.get()] = lightName;

    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Spawn: Camera (real BaseCamera entity + red marker in a group)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::spawnCamera() {
    auto* currentCam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera());
    if (!currentCam) return;

    // Get current camera's world position/rotation (accounts for group hierarchy)
    auto pos = worldPos(currentCam);
    auto rot = worldRot(currentCam);

    std::string name = "Cam_" + std::to_string(++cameraNameCounter);

    // Create camera (local transform = identity; group handles world positioning)
    auto cam = std::make_shared<moon::entities::BaseCamera>(45.0f, window.aspectRatio(), 0.1f);
    cameras[name] = cam;

    auto marker = makeMarkerCube(models["pyramid"].get(),
        moon::math::vec4{ 1.0f, 0.2f, 0.2f, 1.0f }, 0.1f);

    auto group = std::make_shared<moon::transformational::Group>();
    group->setGlobalTransform(moon::math::mat4::identity());
    group->translation() = moon::math::quat(0.0f, pos);
    // Group rotation stays identity — look rotation goes into camera's local m_rotation.
    // This keeps rotateY's yaw axis (0,0,1) aligned with world "up".
    group->add(marker.get());
    group->add(cam.get());
    cam->rotation() = rot;
    group->update();
    groups[name] = group;

    EditorGroup eg;
    eg.group = group.get();
    eg.pivotCube = marker;
    editorGroups[name] = std::move(eg);

    app.deviceWaitIdle();

    for (auto& [_, graph] : graphics)
        graph->bind(marker->object());

    markerOwners[marker->object()] = { group.get(), name };
    registerNode(NodeKind::Group, name, group.get());
    nodeNames[group.get()] = name;

    // Switch to the newly created camera
    switchToCamera(name);

    requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Switch active camera
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::switchToCamera(const std::string& camName) {
    auto it = cameras.find(camName);
    if (it == cameras.end()) return;
    if (camName == activeCameraName) return;

    // Wait for GPU to finish before switching
    app.deviceWaitIdle();

    activeCameraName = camName;
    graphics["base"]->bind(it->second->camera());

    // Re-record command buffers to bind the new camera's descriptor set
    graphics["base"]->requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirror pre-existing engine groups into the editor hierarchy
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::setupEngineGroups() {
    // Extracts a node from hierarchyRoots by name, returns it (or nullopt if not found)
    auto extractNode = [&](const std::string& name) -> std::optional<HierarchyNode> {
        for (auto it = hierarchyRoots.begin(); it != hierarchyRoots.end(); ++it) {
            if (it->name == name) {
                HierarchyNode n = *it;
                hierarchyRoots.erase(it);
                return n;
            }
        }
        return std::nullopt;
    };

    // UFO group: object "ufo" + atomic light sub-unit "ufoLight"
    if (auto grpIt = groups.find("ufo"); grpIt != groups.end()) {
        EditorGroup eg;
        eg.group = grpIt->second.get();
        if (auto n = extractNode("ufo")) eg.children.push_back(*n);
        // ufoLight is an atomic light unit (already an EditorGroup from createLights)
        eg.children.push_back({ NodeKind::Group, "ufoLight", groups["ufoLight"].get() });
        editorGroups["ufo"] = std::move(eg);
        nodeNames[grpIt->second.get()] = "ufo";
        hierarchyRoots.push_back({ NodeKind::Group, "ufo", grpIt->second.get() });
    }

    // Sun: already set up as atomic light unit in createLights().
    // Just remove its flat node from hierarchyRoots (registerNode added it as Group)
    // — it's already there, no extra work needed.
}

// ─────────────────────────────────────────────────────────────────────────────
// Delete a node (object, light group, or user group) from the scene entirely
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::onDeleteNode(const std::string& name) {
    // Don't delete locked nodes or the active camera
    if (editorState.lockedNodes.count(name)) return;
    if (cameras.count(name) && name == activeCameraName) return;

    app.deviceWaitIdle();

    // Helper: remove a hierarchy root by name
    auto removeRoot = [&](const std::string& n) {
        hierarchyRoots.erase(
            std::remove_if(hierarchyRoots.begin(), hierarchyRoots.end(),
                [&](const HierarchyNode& h) { return h.name == n; }),
            hierarchyRoots.end());
    };

    // Clear selection if the deleted node is selected
    for (auto it = editorState.selection.begin(); it != editorState.selection.end(); ) {
        if (nodeNames.count(*it) && nodeNames[*it] == name)
            it = editorState.selection.erase(it);
        else ++it;
    }
    if (editorState.selectedName == name) {
        editorState.selectedObject = nullptr;
        editorState.selectedName.clear();
    }

    // ── Delete an EditorGroup (light group or user group) ────────────────────
    auto egIt = editorGroups.find(name);
    if (egIt != editorGroups.end()) {
        auto& eg = egIt->second;

        // Remove light entity from renderer
        if (eg.lightEntity) {
            if (auto* dl = dynamic_cast<moon::entities::DirectionalLight*>(eg.lightEntity)) {
                baseGraphics()->remove(dl->light());
                if (sunLight.get() == dl) sunLight.reset();
            } else if (auto* sl = dynamic_cast<moon::entities::SpotLight*>(eg.lightEntity)) {
                baseGraphics()->remove(sl->light());
                // Remove from lightSources
                lightSources.erase(
                    std::remove_if(lightSources.begin(), lightSources.end(),
                        [sl](const auto& p) { return p.get() == sl; }),
                    lightSources.end());
                // Remove from groupSpotLights
                for (auto& [_, vec] : groupSpotLights)
                    vec.erase(std::remove(vec.begin(), vec.end(), sl), vec.end());
            } else if (auto* iso = dynamic_cast<moon::entities::IsotropicLight*>(eg.lightEntity)) {
                for (auto* l : iso->getLights())
                    baseGraphics()->remove(l);
            } else if (auto* pl = dynamic_cast<moon::entities::PointLight*>(eg.lightEntity)) {
                baseGraphics()->remove(pl->light());
                lightSources.erase(
                    std::remove_if(lightSources.begin(), lightSources.end(),
                        [pl](const auto& p) { return p.get() == pl; }),
                    lightSources.end());
                pointLightPtrs.erase(
                    std::remove(pointLightPtrs.begin(), pointLightPtrs.end(), pl),
                    pointLightPtrs.end());
            }
        }

        // Remove pivot cube from renderer
        if (eg.pivotCube) {
            markerOwners.erase(eg.pivotCube->object());
            baseGraphics()->remove(eg.pivotCube->object());
        }

        // Recursively delete children that are also EditorGroups
        for (const auto& child : eg.children) {
            if (child.kind == NodeKind::Group && editorGroups.count(child.name))
                onDeleteNode(child.name);
        }

        // Remove children that are objects from the renderer
        for (const auto& child : eg.children) {
            if (child.kind == NodeKind::Object) {
                if (auto* obj = dynamic_cast<moon::transformational::Object*>(child.ptr))
                    baseGraphics()->remove(static_cast<moon::entities::BaseObject*>(obj)->object());
                objects.erase(child.name);
                objectModelKeys.erase(child.name);
            }
        }

        // Remove from parent group if nested
        for (auto& [pname, peg] : editorGroups) {
            if (pname == name) continue;
            auto& ch = peg.children;
            ch.erase(std::remove_if(ch.begin(), ch.end(),
                [&](const HierarchyNode& h) { return h.name == name; }), ch.end());
            if (peg.group) {
                auto grpIt = groups.find(name);
                if (grpIt != groups.end())
                    dynamic_cast<moon::transformational::Group*>(peg.group)->remove(grpIt->second.get());
            }
        }

        if (eg.group) nodeNames.erase(eg.group);
        editorGroups.erase(egIt);
        removeRoot(name);
        groups.erase(name);

        syncOutlines();
        requestUpdate();
        return;
    }

    // ── Delete a standalone object ───────────────────────────────────────────
    auto objIt = objects.find(name);
    if (objIt != objects.end()) {
        baseGraphics()->remove(
            static_cast<moon::entities::BaseObject*>(objIt->second.get())->object());
        nodeNames.erase(objIt->second.get());
        objects.erase(objIt);
        objectModelKeys.erase(name);
        removeRoot(name);
        syncOutlines();
        requestUpdate();
        return;
    }

    // ── Delete a skybox ──────────────────────────────────────────────────────
    auto skyIt = skyboxObjects.find(name);
    if (skyIt != skyboxObjects.end()) {
        baseGraphics()->remove(
            static_cast<moon::entities::SkyboxObject*>(skyIt->second.get())->object());
        nodeNames.erase(skyIt->second.get());
        skyboxObjects.erase(skyIt);
        skyboxTexturePaths.erase(name);
        removeRoot(name);
        requestUpdate();
        return;
    }

    // ── Delete a camera ──────────────────────────────────────────────────────
    if (cameras.count(name) && name != activeCameraName) {
        // Remove its group
        auto grpIt = groups.find(name);
        if (grpIt != groups.end()) {
            auto egIt2 = editorGroups.find(name);
            if (egIt2 != editorGroups.end()) {
                if (egIt2->second.pivotCube) {
                    markerOwners.erase(egIt2->second.pivotCube->object());
                    baseGraphics()->remove(egIt2->second.pivotCube->object());
                }
                nodeNames.erase(egIt2->second.group);
                editorGroups.erase(egIt2);
            }
            groups.erase(grpIt);
        }
        cameras.erase(name);
        removeRoot(name);
        requestUpdate();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rename a node (object, group, light, camera)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::onRenameNode(const std::string& oldName, const std::string& newName) {
    if (oldName == newName) return;
    if (newName.empty()) return;

    // Helper: rename in hierarchyRoots
    for (auto& n : hierarchyRoots)
        if (n.name == oldName) n.name = newName;

    // Helper: rename in EditorGroup children
    for (auto& [_, eg] : editorGroups)
        for (auto& child : eg.children)
            if (child.name == oldName) child.name = newName;

    // Update nodeNames (ptr → name)
    for (auto& [ptr, nm] : nodeNames)
        if (nm == oldName) nm = newName;

    // Rename in maps (move value to new key)
    auto moveKey = [](auto& map, const std::string& from, const std::string& to) {
        auto it = map.find(from);
        if (it == map.end()) return;
        map[to] = std::move(it->second);
        map.erase(it);
    };

    moveKey(objects, oldName, newName);
    moveKey(objectModelKeys, oldName, newName);
    moveKey(skyboxObjects, oldName, newName);
    moveKey(skyboxTexturePaths, oldName, newName);
    moveKey(cameras, oldName, newName);
    moveKey(groups, oldName, newName);
    moveKey(editorGroups, oldName, newName);
    moveKey(groupSpotLights, oldName, newName);
    moveKey(editorState.nodeVisible, oldName, newName);
    moveKey(editorState.savedLightPower, oldName, newName);

    // Update locked set
    if (editorState.lockedNodes.erase(oldName))
        editorState.lockedNodes.insert(newName);

    // Update selected name
    if (editorState.selectedName == oldName)
        editorState.selectedName = newName;

    // Update active camera name
    if (activeCameraName == oldName)
        activeCameraName = newName;

    // Update markerOwners (value contains name)
    for (auto& [_, pair] : markerOwners)
        if (pair.second == oldName) pair.second = newName;
}

void sceneEditor::onReorderNode(const std::string& srcName, const std::string& dstName,
                                const std::string& parentGroup, bool insertAfter) {
    std::vector<HierarchyNode>* vec = nullptr;
    if (parentGroup.empty()) {
        vec = &hierarchyRoots;
    } else {
        auto it = editorGroups.find(parentGroup);
        if (it == editorGroups.end()) return;
        vec = &it->second.children;
    }

    int srcIdx = -1, dstIdx = -1;
    for (int i = 0; i < (int)vec->size(); i++) {
        if ((*vec)[i].name == srcName) srcIdx = i;
        if ((*vec)[i].name == dstName) dstIdx = i;
    }
    if (srcIdx < 0 || dstIdx < 0 || srcIdx == dstIdx) return;

    HierarchyNode node = std::move((*vec)[srcIdx]);
    vec->erase(vec->begin() + srcIdx);
    if (srcIdx < dstIdx) dstIdx--;
    if (insertAfter) dstIdx++;
    if (dstIdx > (int)vec->size()) dstIdx = (int)vec->size();
    vec->insert(vec->begin() + dstIdx, std::move(node));
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene JSON load
// ─────────────────────────────────────────────────────────────────────────────

static void applyRotations(moon::transformational::Transformational& obj, const json& rotations) {
    for (const auto& r : rotations) {
        if (r.contains("quat")) {
            auto q = r["quat"];
            obj.rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                        q[2].get<float>(), q[3].get<float>()));
        } else if (r.contains("angle")) {
            auto ax = r["axis"];
            obj.rotate(moon::math::radians(r["angle"].get<float>()),
                       moon::math::vec3(ax[0].get<float>(), ax[1].get<float>(), ax[2].get<float>()));
        }
    }
}

void sceneEditor::loadSceneFromJson(const std::filesystem::path& jsonPath) {
    std::ifstream file(jsonPath);
    if (!file.is_open()) return;
    json scene = json::parse(file);
    file.close();

    const auto rc = app.getResourceCount();

    // ── Utility models (always needed for markers) ───────────────────────────
    const auto modelPath = ExternalPath / "dependences/model";
    models["cube"]    = std::make_shared<moon::models::PlyModel>(modelPath / "ply/cube.ply");
    models["pyramid"] = std::make_shared<moon::models::PlyModel>(modelPath / "ply/pyramid.ply");
    modelPaths["cube"]    = "dependences/model/ply/cube.ply";
    modelPaths["pyramid"] = "dependences/model/ply/pyramid.ply";

    // ── Camera ───────────────────────────────────────────────────────────────
    if (scene.contains("camera")) {
        auto& cam = scene["camera"];
        float fov = cam.value("fov", 45.0f);
        auto pos  = cam["position"];
        cameras["base"] = std::make_shared<moon::entities::BaseCamera>(fov, window.aspectRatio(), 0.1f);
        cameras["base"]->translate({ pos[0].get<float>(), pos[1].get<float>(), pos[2].get<float>() });
        if (cam.contains("rotation")) {
            auto& q = cam["rotation"];
            cameras["base"]->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                     q[2].get<float>(), q[3].get<float>()));
        }
        registerNode(NodeKind::Camera, "base", cameras["base"].get());
    }

    // ── Pass 1: Load all models ─────────────────────────────────────────────
    if (scene.contains("objects")) {
        for (const auto& obj : scene["objects"]) {
            std::string name = obj["name"].get<std::string>();
            if (!obj.contains("path")) continue;  // skip objects saved without a model path
            std::string path = obj["path"].get<std::string>();
            bool animated    = obj.value("animated", false);
            int instances    = obj.value("instances", 1);

            if (models.find(name) == models.end()) {
                auto fullPath = ExternalPath / path;
                auto ext = fullPath.extension().string();
                if (ext == ".ply") {
                    models[name] = std::make_shared<moon::models::PlyModel>(fullPath);
                } else {
                    uint32_t ic = animated ? std::max(1, instances) * rc : 1;
                    models[name] = std::make_shared<moon::models::GltfModel>(fullPath, ic);
                }
                modelPaths[name] = path;
            }
        }
    }

    // ── Upload all models to GPU ─────────────────────────────────────────────
    for (auto& [_, m] : models)
        graphics["base"]->create(m.get());

    // ── Pass 2: Create objects from loaded models ────────────────────────────
    if (scene.contains("objects")) {
        for (const auto& obj : scene["objects"]) {
            std::string name = obj["name"].get<std::string>();
            if (!obj.contains("path")) continue;  // skip objects saved without a model path
            bool animated    = obj.value("animated", false);

            uint32_t objInstances = animated ? rc : 1;
            auto object = std::make_shared<moon::entities::BaseObject>(models[name].get(), 0, objInstances);

            // Apply scale
            if (obj.contains("scale")) {
                auto& s = obj["scale"];
                object->scale(moon::math::vec3(s[0].get<float>(), s[1].get<float>(), s[2].get<float>()));
            }

            // Apply rotations
            if (obj.contains("rotations"))
                applyRotations(*object, obj["rotations"]);
            else if (obj.contains("rotation")) {
                auto& q = obj["rotation"];
                object->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                q[2].get<float>(), q[3].get<float>()));
            }

            // Apply translation
            if (obj.contains("position")) {
                auto& p = obj["position"];
                object->translate({ p[0].get<float>(), p[1].get<float>(), p[2].get<float>() });
            }

            // Play animation if requested
            if (obj.contains("playAnimation")) {
                auto* baseObj = static_cast<moon::entities::BaseObject*>(object.get());
                baseObj->animation.play(obj["playAnimation"].get<int>());
            }

            // Handle group
            if (obj.contains("group")) {
                auto& grp = obj["group"];
                std::string grpName = grp["name"].get<std::string>();
                if (groups.find(grpName) == groups.end()) {
                    groups[grpName] = std::make_shared<moon::transformational::Group>();
                    groups[grpName]->setGlobalTransform(moon::math::mat4::identity());
                }
                groups[grpName]->add(object.get());
                if (grp.contains("position")) {
                    auto& gp = grp["position"];
                    groups[grpName]->translate({ gp[0].get<float>(), gp[1].get<float>(), gp[2].get<float>() });
                }
                if (grp.contains("rotation")) {
                    auto& gq = grp["rotation"];
                    groups[grpName]->rotate(moon::math::quat(gq[0].get<float>(), gq[1].get<float>(),
                                                             gq[2].get<float>(), gq[3].get<float>()));
                }
            }

            objects[name] = object;
            objectModelKeys[name] = name;
        }
    }

    // ── Skybox ───────────────────────────────────────────────────────────────
    if (scene.contains("skybox")) {
        auto& sky = scene["skybox"];
        std::string skyName = sky.value("name", std::string("skybox"));
        float skyScale = sky.value("scale", 200.0f);

        moon::utils::vkDefault::Paths texPaths;
        std::vector<std::string> relPaths;
        for (const auto& t : sky["textures"]) {
            std::string relPath = t.get<std::string>();
            texPaths.push_back(ExternalPath / relPath);
            relPaths.push_back(relPath);
        }
        skyboxTexturePaths[skyName] = relPaths;

        skyboxObjects[skyName] = std::make_shared<moon::entities::SkyboxObject>(texPaths);
        skyboxObjects[skyName]->scale({ skyScale, skyScale, skyScale });
    }

    // ── Bind objects to renderer ─────────────────────────────────────────────
    for (auto& [_, graph] : graphics) {
        for (auto& [_, obj] : objects)       graph->bind(obj->object());
        for (auto& [_, obj] : skyboxObjects) graph->bind(obj->object());
    }

    // ── Register hierarchy nodes for objects ─────────────────────────────────
    for (auto& [name, obj] : objects)
        registerNode(NodeKind::Object, name, obj.get());
    for (auto& [name, obj] : skyboxObjects)
        registerNode(NodeKind::SkyboxObject, name, obj.get());

    // ── Lights ───────────────────────────────────────────────────────────────
    if (scene.contains("lights")) {
        for (const auto& lt : scene["lights"]) {
            std::string name     = lt["name"].get<std::string>();
            std::string type     = lt["type"].get<std::string>();
            std::string parent   = lt.value("parent", std::string());
            auto col = lt["color"];
            moon::math::vec4 color(col[0].get<float>(), col[1].get<float>(),
                                   col[2].get<float>(), col[3].get<float>());

            // Marker
            std::string markerModel = lt.value("markerModel", std::string("cube"));
            float markerSize = lt.value("markerSize", 0.08f);
            moon::math::vec4 markerColor = color;
            if (lt.contains("markerColor")) {
                auto& mc = lt["markerColor"];
                markerColor = moon::math::vec4(mc[0].get<float>(), mc[1].get<float>(),
                                               mc[2].get<float>(), mc[3].get<float>());
            }

            if (type == "spot") {
                auto light = std::make_shared<moon::entities::SpotLight>(
                    moon::entities::SpotLight::Coloring(color),
                    moon::entities::SpotLight::Props{
                        lt.value("shadow", true),
                        lt.value("scattering", false),
                        lt.value("drop", 1.0f),
                        lt.value("power", 10.0f),
                        lt.value("innerFraction", 1.0f),
                        lt.value("exponent", 4.0f)
                    });
                lightSources.push_back(light);

                if (!parent.empty() && groups.count(parent))
                    groupSpotLights[parent].push_back(light.get());

                auto marker = makeMarkerCube(models[markerModel].get(), markerColor, markerSize);

                auto group = std::make_shared<moon::transformational::Group>();
                group->setGlobalTransform(moon::math::mat4::identity());
                group->add(light.get());
                group->add(marker.get());
                groups[name] = group;

                if (!parent.empty() && groups.count(parent))
                    groups[parent]->add(group.get());

                if (lt.contains("position")) {
                    auto& p = lt["position"];
                    group->translate({ p[0].get<float>(), p[1].get<float>(), p[2].get<float>() });
                }
                if (lt.contains("rotation")) {
                    auto& q = lt["rotation"];
                    group->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                   q[2].get<float>(), q[3].get<float>()));
                }

                EditorGroup eg;
                eg.group = group.get();
                eg.pivotCube = marker;
                eg.lightEntity = light.get();
                editorGroups[name] = std::move(eg);
                nodeNames[group.get()] = name;

                for (auto& [_, graph] : graphics) {
                    graph->bind(light->light());
                    graph->bind(marker->object());
                }
                markerOwners[marker->object()] = { group.get(), name };

                // Only register as top-level if no parent
                if (parent.empty())
                    registerNode(NodeKind::Group, name, group.get());

            } else if (type == "directional") {
                sunLight = std::make_shared<moon::entities::DirectionalLight>(color,
                    moon::entities::DirectionalLight::Props{
                        lt.value("shadow", true),
                        lt.value("power", 10.0f),
                        lt.value("drop", 0.0f),
                        lt.value("width", 100.0f),
                        lt.value("height", 100.0f),
                        lt.value("far", 200.0f)
                    });

                auto marker = makeMarkerCube(models[markerModel].get(), markerColor, markerSize);

                groups[name] = std::make_shared<moon::transformational::Group>();
                groups[name]->setGlobalTransform(moon::math::mat4::identity());
                groups[name]->add(sunLight.get());
                groups[name]->add(marker.get());

                if (lt.contains("position")) {
                    auto& p = lt["position"];
                    groups[name]->translate({ p[0].get<float>(), p[1].get<float>(), p[2].get<float>() });
                }
                if (lt.contains("rotation")) {
                    auto& q = lt["rotation"];
                    groups[name]->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                          q[2].get<float>(), q[3].get<float>()));
                }

                EditorGroup eg;
                eg.group = groups[name].get();
                eg.pivotCube = marker;
                eg.lightEntity = sunLight.get();
                editorGroups[name] = std::move(eg);
                nodeNames[groups[name].get()] = name;

                for (auto& [_, graph] : graphics) {
                    graph->bind(sunLight->light());
                    graph->bind(marker->object());
                }
                markerOwners[marker->object()] = { groups[name].get(), name };
                registerNode(NodeKind::Group, name, groups[name].get());

            } else if (type == "point") {
                auto light = std::make_shared<moon::entities::PointLight>(
                    moon::entities::PointLight::Props{
                        color,
                        lt.value("radius", 10.0f),
                        lt.value("power", 10.0f),
                        lt.value("drop", 1.0f)
                    });
                pointLightPtrs.push_back(light.get());
                lightSources.push_back(light);

                auto marker = makeMarkerCube(models[markerModel].get(), markerColor, markerSize);

                auto group = std::make_shared<moon::transformational::Group>();
                group->setGlobalTransform(moon::math::mat4::identity());
                group->add(light.get());
                group->add(marker.get());

                if (lt.contains("position")) {
                    auto& p = lt["position"];
                    group->translate({ p[0].get<float>(), p[1].get<float>(), p[2].get<float>() });
                }
                if (lt.contains("rotation")) {
                    auto& q = lt["rotation"];
                    group->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                   q[2].get<float>(), q[3].get<float>()));
                }

                groups[name] = group;

                EditorGroup eg;
                eg.group = group.get();
                eg.pivotCube = marker;
                eg.lightEntity = light.get();
                editorGroups[name] = std::move(eg);
                nodeNames[group.get()] = name;

                for (auto& [_, graph] : graphics) {
                    graph->bind(light->light());
                    graph->bind(marker->object());
                }
                markerOwners[marker->object()] = { group.get(), name };
                registerNode(NodeKind::Group, name, group.get());

            } else if (type == "isotropic") {
                auto isoLight = std::make_shared<moon::entities::IsotropicLight>(
                    moon::entities::IsotropicLight::Props{
                        color,
                        lt.value("radius", 100.0f),
                        lt.value("shadow", true),
                        lt.value("scattering", false),
                        lt.value("drop", 1.0f),
                        lt.value("power", 10.0f),
                        lt.value("innerFraction", 1.0f),
                        lt.value("exponent", 4.0f)
                    });
                isoLight->setGlobalTransform(moon::math::mat4::identity());

                auto marker = makeMarkerCube(models[markerModel].get(), markerColor, markerSize);
                isoLight->add(marker.get());

                if (lt.contains("position")) {
                    auto& p = lt["position"];
                    isoLight->translate({ p[0].get<float>(), p[1].get<float>(), p[2].get<float>() });
                }
                if (lt.contains("rotation")) {
                    auto& q = lt["rotation"];
                    isoLight->rotate(moon::math::quat(q[0].get<float>(), q[1].get<float>(),
                                                      q[2].get<float>(), q[3].get<float>()));
                }

                groups[name] = isoLight;

                EditorGroup eg;
                eg.group = isoLight.get();
                eg.pivotCube = marker;
                eg.lightEntity = isoLight.get();
                editorGroups[name] = std::move(eg);
                nodeNames[isoLight.get()] = name;

                for (auto& [_, graph] : graphics) {
                    for (auto* l : isoLight->getLights())
                        graph->bind(l);
                    graph->bind(marker->object());
                }
                markerOwners[marker->object()] = { isoLight.get(), name };
                registerNode(NodeKind::Group, name, isoLight.get());
            }
        }
    }

    // ── Engine groups (parent-child wiring) ──────────────────────────────────
    // For each object that declared a group, wire it into the editor hierarchy
    if (scene.contains("objects")) {
        for (const auto& obj : scene["objects"]) {
            if (!obj.contains("group")) continue;
            std::string grpName = obj["group"]["name"].get<std::string>();
            std::string objName = obj["name"].get<std::string>();
            auto grpIt = groups.find(grpName);
            if (grpIt == groups.end()) continue;

            // Build EditorGroup if not yet created
            if (editorGroups.find(grpName) == editorGroups.end()) {
                EditorGroup eg;
                eg.group = grpIt->second.get();
                editorGroups[grpName] = std::move(eg);
                nodeNames[grpIt->second.get()] = grpName;
            }

            // Move object from top-level roots into the group
            auto& eg = editorGroups[grpName];
            for (auto it = hierarchyRoots.begin(); it != hierarchyRoots.end(); ++it) {
                if (it->name == objName) {
                    eg.children.push_back(*it);
                    hierarchyRoots.erase(it);
                    break;
                }
            }

            // Add child light groups (like ufoLight inside ufo)
            for (const auto& [lgName, lgEg] : editorGroups) {
                if (lgName == grpName) continue;
                // Check if this light group is parented to our group
                if (scene.contains("lights")) {
                    for (const auto& lt : scene["lights"]) {
                        if (lt.value("name", std::string()) == lgName &&
                            lt.value("parent", std::string()) == grpName) {
                            eg.children.push_back({ NodeKind::Group, lgName, groups[lgName].get() });
                        }
                    }
                }
            }

            // Replace flat object node with group node in hierarchy
            bool found = false;
            for (auto& n : hierarchyRoots) {
                if (n.name == grpName) { found = true; break; }
            }
            if (!found)
                hierarchyRoots.push_back({ NodeKind::Group, grpName, grpIt->second.get() });
        }
    }

    // ── Locked nodes ─────────────────────────────────────────────────────────
    if (scene.contains("locked")) {
        for (const auto& name : scene["locked"])
            editorState.lockedNodes.insert(name.get<std::string>());
    }

    // ── Initialize name counters to avoid collisions with loaded entities ────
    auto maxSuffix = [](const auto& map, const std::string& prefix) -> int {
        int maxVal = 0;
        for (const auto& [name, _] : map) {
            if (name.rfind(prefix, 0) == 0) { // starts with prefix
                try { maxVal = std::max(maxVal, std::stoi(name.substr(prefix.size()))); }
                catch (...) {}
            }
        }
        return maxVal;
    };
    spotLightNameCounter  = maxSuffix(editorGroups, "Spot_");
    dirLightNameCounter   = maxSuffix(editorGroups, "Dir_");
    isoLightNameCounter   = maxSuffix(editorGroups, "Iso_");
    cameraNameCounter     = maxSuffix(cameras,      "Cam_");
    objectNameCounter     = maxSuffix(objects,       "_");  // fallback: scan all
    groupNameCounter      = maxSuffix(editorGroups,  "Group_");

    // Also scan object names for "_dyn_N" pattern
    for (const auto& [name, _] : objects) {
        auto pos = name.rfind("_dyn_");
        if (pos != std::string::npos) {
            try { objectNameCounter = std::max(objectNameCounter, std::stoi(name.substr(pos + 5))); }
            catch (...) {}
        }
        // Also check for "model_N" pattern (from spawnObject)
        auto lastUnderscore = name.rfind('_');
        if (lastUnderscore != std::string::npos) {
            try { objectNameCounter = std::max(objectNameCounter, std::stoi(name.substr(lastUnderscore + 1))); }
            catch (...) {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene JSON save
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::saveSceneToJson(const std::filesystem::path& jsonPath) const {
    json scene;

    // ── Camera ───────────────────────────────────────────────────────────────
    if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(activeCamera())) {
        // Position comes from the group, rotation is camera-local (for proper yaw/pitch)
        auto pos = worldPos(cam);
        auto rot = cam->rotation();
        scene["camera"] = {
            { "position", { pos[0], pos[1], pos[2] } },
            { "rotation", { rot.re(), rot.im()[0], rot.im()[1], rot.im()[2] } },
            { "fov", 45.0f }
        };
    }

    // ── Skybox ───────────────────────────────────────────────────────────────
    for (const auto& [name, obj] : skyboxObjects) {
        auto it = skyboxTexturePaths.find(name);
        if (it == skyboxTexturePaths.end()) continue;
        auto s = obj->scaling();
        scene["skybox"] = {
            { "name", name },
            { "textures", it->second },
            { "scale", s[0] }
        };
        break; // only one skybox
    }

    // ── Objects ──────────────────────────────────────────────────────────────
    json objArray = json::array();
    for (const auto& [name, obj] : objects) {
        // Skip marker cubes (they belong to editor groups)
        bool isMarker = false;
        for (const auto& [_, eg] : editorGroups)
            if (eg.pivotCube.get() == obj.get()) { isMarker = true; break; }
        if (isMarker) continue;

        json jobj;
        jobj["name"] = name;

        // Find model path — skip objects with no known model path
        auto mkIt = objectModelKeys.find(name);
        if (mkIt != objectModelKeys.end()) {
            auto mpIt = modelPaths.find(mkIt->second);
            if (mpIt != modelPaths.end())
                jobj["path"] = mpIt->second;
            else
                continue; // can't serialize without a model path
        } else {
            continue; // can't serialize without a model key
        }

        // Transform — read from Transformational via dynamic_cast
        auto* tObj = dynamic_cast<moon::transformational::Object*>(obj.get());
        if (tObj) {
            auto pos = tObj->translation().im();
            auto rot = tObj->rotation();
            auto scl = tObj->scaling();
            jobj["position"] = { pos[0], pos[1], pos[2] };
            jobj["rotation"] = { rot.re(), rot.im()[0], rot.im()[1], rot.im()[2] };
            jobj["scale"]    = { scl[0], scl[1], scl[2] };
        }

        // Check if animated
        auto* baseObj = dynamic_cast<moon::entities::BaseObject*>(obj.get());
        int curAnim = baseObj ? baseObj->animation.current() : -1;
        jobj["animated"] = curAnim >= 0;
        if (curAnim >= 0)
            jobj["playAnimation"] = curAnim;

        // Check if in a group
        for (const auto& [grpName, eg] : editorGroups) {
            for (const auto& child : eg.children) {
                if (child.name == name) {
                    auto* grp = dynamic_cast<moon::transformational::Group*>(eg.group);
                    if (grp) {
                        auto gpos = grp->translation().im();
                        auto grot = grp->rotation();
                        jobj["group"] = {
                            { "name", grpName },
                            { "position", { gpos[0], gpos[1], gpos[2] } },
                            { "rotation", { grot.re(), grot.im()[0], grot.im()[1], grot.im()[2] } }
                        };
                    }
                    break;
                }
            }
        }

        objArray.push_back(jobj);
    }
    scene["objects"] = objArray;

    // ── Lights ───────────────────────────────────────────────────────────────
    json lightArray = json::array();
    for (const auto& [name, eg] : editorGroups) {
        if (!eg.lightEntity) continue;
        // Skip camera groups
        if (cameras.count(name)) continue;

        json jlt;
        jlt["name"] = name;

        // Determine parent group
        for (const auto& [pname, peg] : editorGroups) {
            if (pname == name) continue;
            for (const auto& child : peg.children) {
                if (child.name == name) {
                    jlt["parent"] = pname;
                    break;
                }
            }
            if (jlt.contains("parent")) break;
        }

        // Position from group
        if (eg.group) {
            auto* grp = dynamic_cast<moon::transformational::Group*>(eg.group);
            if (grp) {
                auto pos = grp->translation().im();
                auto rot = grp->rotation();
                if (pos[0] != 0.0f || pos[1] != 0.0f || pos[2] != 0.0f)
                    jlt["position"] = { pos[0], pos[1], pos[2] };
                if (rot.re() != 1.0f || rot.im()[0] != 0.0f || rot.im()[1] != 0.0f || rot.im()[2] != 0.0f)
                    jlt["rotation"] = { rot.re(), rot.im()[0], rot.im()[1], rot.im()[2] };
            }
        }

        // Type-specific properties
        if (auto* dl = dynamic_cast<moon::entities::DirectionalLight*>(eg.lightEntity)) {
            jlt["type"]   = "directional";
            auto c = dl->getColor();
            jlt["color"]  = { c[0], c[1], c[2], c[3] };
            jlt["shadow"] = dl->getEnableShadow();
            jlt["power"]  = dl->getPower();
            jlt["drop"]   = dl->getDrop();
            jlt["width"]  = dl->getWidth();
            jlt["height"] = dl->getHeight();
            jlt["far"]    = dl->getFar();
        } else if (auto* sl = dynamic_cast<moon::entities::SpotLight*>(eg.lightEntity)) {
            jlt["type"]          = "spot";
            auto c = sl->getColor();
            jlt["color"]         = { c[0], c[1], c[2], c[3] };
            jlt["shadow"]        = sl->getEnableShadow();
            jlt["scattering"]    = sl->getEnableScattering();
            jlt["drop"]          = sl->getDrop();
            jlt["power"]         = sl->getPower();
            jlt["innerFraction"] = sl->getInnerFraction();
            jlt["exponent"]      = sl->getExponent();
        } else if (auto* iso = dynamic_cast<moon::entities::IsotropicLight*>(eg.lightEntity)) {
            jlt["type"]          = "isotropic";
            auto c = iso->getColor();
            jlt["color"]         = { c[0], c[1], c[2], c[3] };
            jlt["shadow"]        = iso->getEnableShadow();
            jlt["scattering"]    = iso->getEnableScattering();
            jlt["drop"]          = iso->getDrop();
            jlt["power"]         = iso->getPower();
            jlt["innerFraction"] = iso->getInnerFraction();
            jlt["exponent"]      = iso->getExponent();
        } else if (auto* pl = dynamic_cast<moon::entities::PointLight*>(eg.lightEntity)) {
            jlt["type"]   = "point";
            auto c = pl->getColor();
            jlt["color"]  = { c[0], c[1], c[2], c[3] };
            jlt["radius"] = pl->getRadius();
            jlt["power"]  = pl->getPower();
            jlt["drop"]   = pl->getDrop();
        }

        // Marker info
        if (eg.pivotCube) {
            auto s = eg.pivotCube->scaling();
            jlt["markerSize"] = s[0];
        }

        lightArray.push_back(jlt);
    }
    scene["lights"] = lightArray;

    // ── Locked ───────────────────────────────────────────────────────────────
    json locked = json::array();
    for (const auto& name : editorState.lockedNodes)
        locked.push_back(name);
    scene["locked"] = locked;

    // ── Write ────────────────────────────────────────────────────────────────
    std::ofstream out(jsonPath);
    out << scene.dump(4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lock mechanism
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::onToggleLock(const std::string& name) {
    auto& locked = editorState.lockedNodes;
    if (locked.count(name)) locked.erase(name);
    else                    locked.insert(name);
}

// ─────────────────────────────────────────────────────────────────────────────
// Light group detection (atomic light+pivot units, like camera groups)
// ─────────────────────────────────────────────────────────────────────────────

bool sceneEditor::isLightGroup(const std::string& name) const {
    if (cameras.count(name)) return false;
    auto it = editorGroups.find(name);
    if (it == editorGroups.end()) return false;
    if (!it->second.pivotCube) return false;
    for (const auto& child : it->second.children)
        if (child.kind != NodeKind::Light) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Light enable/disable (for visibility toggle)
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::setLightEnabled(const std::string& groupName, bool enabled) {
    auto egIt = editorGroups.find(groupName);
    if (egIt == editorGroups.end()) return;

    auto* ptr = egIt->second.lightEntity;
    if (!ptr) return;

    if (auto* dl = dynamic_cast<moon::entities::DirectionalLight*>(ptr)) {
        dl->setEnable(enabled);
    } else if (auto* sl = dynamic_cast<moon::entities::SpotLight*>(ptr)) {
        if (!enabled) {
            editorState.savedLightPower[groupName] = sl->getPower();
            sl->setPower(0.0f);
        } else if (editorState.savedLightPower.count(groupName)) {
            sl->setPower(editorState.savedLightPower[groupName]);
        }
    } else if (auto* pl = dynamic_cast<moon::entities::PointLight*>(ptr)) {
        if (!enabled) {
            editorState.savedLightPower[groupName] = pl->getPower();
            pl->setPower(0.0f);
        } else if (editorState.savedLightPower.count(groupName)) {
            pl->setPower(editorState.savedLightPower[groupName]);
        }
    }

    if (auto* g = baseGraphics()) g->requestUpdate();
}

// ─────────────────────────────────────────────────────────────────────────────
// Request GPU update
// ─────────────────────────────────────────────────────────────────────────────

void sceneEditor::requestUpdate() {
    if (auto it = graphics.find("base"); it != graphics.end()) {
        it->second->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
        it->second->requestUpdate(moon::deferredGraphics::Names::Shadow::name);
        it->second->requestUpdate(moon::deferredGraphics::Names::BoundingBox::name);
        it->second->requestUpdate(moon::deferredGraphics::Names::Scattering::name);
    }
}
