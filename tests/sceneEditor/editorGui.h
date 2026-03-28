#ifndef EDITOR_GUI_H
#define EDITOR_GUI_H

#include "sceneEditor.h"

#include <models/plymodel.h>   // must come before testSceneGui.h (which uses moon::models::PlyModel)

#include "gui.h"
#include "testSceneGui.h"

#include <entities/baseCamera.h>
#include <entities/baseObject.h>
#include <entities/skyboxObject.h>
#include <entities/spotLight.h>
#include <entities/pointLight.h>
#include <entities/directionalLight.h>
#include <transformationals/group.h>
#include <deferredGraphics/deferredGraphics.h>
#include <interfaces/object.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>

namespace editor::gui {

// ─────────────────────────────────────────────────────────────────────────────
// Non-owning shared_ptr shim: lets raw-pointer gfx be passed to helpers
// that take std::shared_ptr<DeferredGraphics>
// ─────────────────────────────────────────────────────────────────────────────
inline std::shared_ptr<moon::deferredGraphics::DeferredGraphics>
noOwn(moon::deferredGraphics::DeferredGraphics* p) {
    return { p, [](moon::deferredGraphics::DeferredGraphics*){} };
}

// ─────────────────────────────────────────────────────────────────────────────
// Name passes the hierarchy search filter (case-insensitive substring)
// ─────────────────────────────────────────────────────────────────────────────
inline bool passFilter(const char* filter, const std::string& name) {
    if (filter[0] == '\0') return true;
    std::string lower = name, flt = filter;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::transform(flt.begin(), flt.end(), flt.begin(), ::tolower);
    return lower.find(flt) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply visibility to a node
// ─────────────────────────────────────────────────────────────────────────────
inline void applyNodeVisibility(sceneEditor& se, const HierarchyNode& node, bool visible) {
    if (node.kind == NodeKind::Object) {
        if (auto* bo = dynamic_cast<moon::entities::BaseObject*>(
                dynamic_cast<moon::transformational::Object*>(node.ptr)))
            bo->setEnable(visible);
    } else if (node.kind == NodeKind::SkyboxObject) {
        if (auto* sky = dynamic_cast<moon::entities::SkyboxObject*>(
                dynamic_cast<moon::transformational::Object*>(node.ptr)))
            sky->object()->objectMask().set(moon::interfaces::ObjectProperty::enable, visible);
        if (auto* g = se.baseGraphics())
            g->requestUpdate(moon::deferredGraphics::Names::Skybox::name);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rename state for hierarchy nodes (shared across recursive calls)
// ─────────────────────────────────────────────────────────────────────────────
struct HierarchyRenameState {
    std::string nodeName;       // node currently being renamed (empty = idle)
    char buffer[256]{};
    bool focusSet{false};
};
inline HierarchyRenameState g_rename;

// Drag-and-drop payload for hierarchy reordering
struct HierarchyDragPayload {
    char nodeName[256]{};
    char parentGroup[256]{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Recursive hierarchy node drawing (forward-declared for mutual recursion)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawHierarchyNode(sceneEditor& se, const HierarchyNode& node, const char* flt,
                              bool isTopLevel = false, const std::string& parentGroup = "");

inline void drawHierarchyNode(sceneEditor& se, const HierarchyNode& node, const char* flt,
                              bool isTopLevel, const std::string& parentGroup) {
    auto& st = se.state();
    if (!passFilter(flt, node.name)) return;

    // Badge + colour per kind (order must match NodeKind enum)
    static constexpr struct { const char* badge; float r,g,b; } styles[] = {
        {"[OBJ]", 0.4f,0.8f,1.0f},   // Object
        {"[SKY]", 0.8f,0.5f,1.0f},    // SkyboxObject
        {"[CAM]", 1.0f,0.85f,0.2f},   // Camera
        {"[LGT]", 1.0f,0.95f,0.3f},   // Light
        {"[GRP]", 0.9f,0.6f,0.2f},    // Group
    };
    const auto& s = styles[static_cast<int>(node.kind)];
    ImVec4 badgeCol{ s.r, s.g, s.b, 1.0f };

    bool inSel    = st.selection.count(node.ptr) > 0;
    bool isPrimary = inSel && (st.selectedName == node.name);
    bool ctrlHeld = ImGui::GetIO().KeyCtrl;
    bool locked   = isTopLevel && se.isNodeLocked(node.name);

    // ── Group: tree node with children ───────────────────────────────────────
    if (node.kind == NodeKind::Group) {
        auto egIt = se.getEditorGroups().find(node.name);
        if (egIt == se.getEditorGroups().end()) return;

        bool isCamGroup  = se.isCameraGroup(node.name);
        bool isLightGrp  = se.isLightGroup(node.name);
        bool isAtomicGrp = isCamGroup || isLightGrp;

        // Badge colour: active camera = green, inactive camera = red, light = gold
        if (isCamGroup && node.name == se.getActiveCameraName())
            badgeCol = { 0.3f, 1.0f, 0.3f, 1.0f };
        else if (isCamGroup)
            badgeCol = { 1.0f, 0.3f, 0.3f, 1.0f };
        else if (isLightGrp)
            badgeCol = { 1.0f, 0.9f, 0.3f, 1.0f };

        const char* badge = isCamGroup ? "[CAM]" : isLightGrp ? "[LGT]" : s.badge;

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth |
                                   ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_DefaultOpen;
        if (inSel) flags |= ImGuiTreeNodeFlags_Selected;

        // Persistent highlight for active camera
        bool activeCamHighlight = isCamGroup && node.name == se.getActiveCameraName() && !inSel;
        int stylesPushed = 0;
        if (inSel) {
            ImGui::PushStyleColor(ImGuiCol_Header,        {0.26f,0.59f,0.98f,0.35f});
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.26f,0.59f,0.98f,0.55f});
            stylesPushed = 2;
        } else if (activeCamHighlight) {
            ImGui::PushStyleColor(ImGuiCol_Header,        {0.1f,0.4f,0.1f,0.30f});
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.1f,0.4f,0.1f,0.50f});
            flags |= ImGuiTreeNodeFlags_Selected;
            stylesPushed = 2;
        }

        ImGui::PushID(node.name.c_str());

        // Light group visibility checkbox (controls light enable)
        if (isLightGrp) {
            if (!st.nodeVisible.count(node.name)) st.nodeVisible[node.name] = true;
            bool& vis = st.nodeVisible[node.name];
            bool prev = vis;
            ImGui::Checkbox("##gvis", &vis);
            if (vis != prev) {
                se.setLightEnabled(node.name, vis);
                se.requestUpdate();
            }
            ImGui::SameLine(0, 4);
        }

        bool open = ImGui::TreeNodeEx("##g", flags);
        bool nodeClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left)
                           && !ImGui::IsItemToggledOpen();

        // ── Drag & Drop (reorder) ────────────────────────────────────────
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            HierarchyDragPayload pl{};
            snprintf(pl.nodeName, sizeof(pl.nodeName), "%s", node.name.c_str());
            snprintf(pl.parentGroup, sizeof(pl.parentGroup), "%s", parentGroup.c_str());
            ImGui::SetDragDropPayload("HIER_NODE", &pl, sizeof(pl));
            ImGui::TextColored(badgeCol, "%s", badge);
            ImGui::SameLine(0, 6);
            ImGui::TextUnformatted(node.name.c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("HIER_NODE")) {
                auto* data = (const HierarchyDragPayload*)p->Data;
                if (std::string(data->parentGroup) == parentGroup) {
                    bool after = ImGui::GetMousePos().y > (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) * 0.5f;
                    se.onReorderNode(data->nodeName, node.name, parentGroup, after);
                }
            }
            ImGui::EndDragDropTarget();
        }

        ImGui::SameLine(0, 4);
        ImGui::TextColored(badgeCol, "%s", badge);
        ImGui::SameLine(0, 6);
        bool nodeModified = false;   // set after rename/delete to skip invalidated iterators
        if (g_rename.nodeName == node.name) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (!g_rename.focusSet) { ImGui::SetKeyboardFocusHere(); g_rename.focusSet = true; }
            bool enter = ImGui::InputText("##ren", g_rename.buffer, sizeof(g_rename.buffer),
                                           ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
            bool lost  = ImGui::IsItemDeactivated();
            if (enter || lost) {
                std::string nn(g_rename.buffer);
                if (!nn.empty() && nn != node.name) { se.onRenameNode(node.name, nn); nodeModified = true; }
                g_rename.nodeName.clear();
            }
        } else {
            ImGui::TextUnformatted(node.name.c_str());
        }

        // Lock indicator
        if (locked && g_rename.nodeName != node.name) {
            ImGui::SameLine(0, 4);
            ImGui::TextColored({1.0f, 0.5f, 0.0f, 0.7f}, "[L]");
        }

        if (stylesPushed) ImGui::PopStyleColor(stylesPushed);

        // Right-click context menu
        if (!nodeModified && ImGui::BeginPopupContextItem("##grpctx")) {
            if (isTopLevel) {
                bool lk = se.isNodeLocked(node.name);
                if (ImGui::MenuItem(lk ? "Unlock" : "Lock"))
                    se.onToggleLock(node.name);
            }
            if (ImGui::MenuItem("Rename")) {
                g_rename.nodeName = node.name;
                snprintf(g_rename.buffer, sizeof(g_rename.buffer), "%s", node.name.c_str());
                g_rename.focusSet = false;
            }
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 0.3f, 0.3f, 1.0f});
            if (ImGui::MenuItem("Delete")) { se.onDeleteNode(node.name); nodeModified = true; }
            ImGui::PopStyleColor();
            ImGui::EndPopup();
        }

        if (!nodeModified && nodeClicked) {
            if (isCamGroup && !ctrlHeld) {
                se.onSwitchToCamera(node.name);
            } else {
                se.onToggleSelection(node.ptr, node.name, ctrlHeld);
            }
        }

        if (open) {
            if (!nodeModified) {
                // Pivot cube row: checkbox to mute + coloured label
                const auto& eg = egIt->second;
                if (eg.pivotCube) {
                    bool pivVis = eg.pivotCube->isEnable();
                    ImGui::PushID("pvchk");
                    if (ImGui::Checkbox("##pvv", &pivVis)) {
                        eg.pivotCube->setEnable(pivVis);
                        se.requestUpdate();
                    }
                    ImGui::PopID();
                    ImGui::SameLine(0, 4);
                    ImGui::TextColored({1.0f, 0.85f, 0.0f, 1.0f}, "[PIV]");
                    ImGui::SameLine(0, 6);
                    if (ImGui::Selectable("Pivot", inSel, ImGuiSelectableFlags_None))
                        se.onToggleSelection(node.ptr, node.name, ctrlHeld);
                }

                const auto children = egIt->second.children; // copy: safe if delete modifies vector
                for (const auto& child : children)
                    drawHierarchyNode(se, child, flt, false, node.name);

                // Disband button only for user-created groups (not atomic light/camera units)
                if (!isAtomicGrp) {
                    ImGui::Separator();
                    ImGui::PushStyleColor(ImGuiCol_Button, {0.55f,0.15f,0.15f,1.0f});
                    if (ImGui::SmallButton("Disband Group")) se.onDisbandGroup(node.name);
                    ImGui::PopStyleColor();
                }
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
        return;
    }

    // ── Regular item ─────────────────────────────────────────────────────────
    ImGui::PushID(node.name.c_str());

    // Visibility checkbox (Object and SkyboxObject)
    bool hasVis = (node.kind == NodeKind::Object || node.kind == NodeKind::SkyboxObject);
    if (hasVis) {
        if (!st.nodeVisible.count(node.name)) st.nodeVisible[node.name] = true;
        bool& vis  = st.nodeVisible[node.name];
        bool  prev = vis;
        ImGui::Checkbox("##vis", &vis);
        if (vis != prev) {
            applyNodeVisibility(se, node, vis);
            se.requestUpdate();
        }
    } else {
        ImGui::BeginDisabled();
        bool dummy = true;
        ImGui::Checkbox("##vis", &dummy);
        ImGui::EndDisabled();
    }
    ImGui::SameLine(0, 4);
    ImGui::TextColored(badgeCol, "%s", s.badge);
    ImGui::SameLine(0, 6);

    bool clicked = false;
    bool nodeModified = false;
    if (g_rename.nodeName == node.name) {
        // ── Inline rename ────────────────────────────────────────────────
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (!g_rename.focusSet) { ImGui::SetKeyboardFocusHere(); g_rename.focusSet = true; }
        bool enter = ImGui::InputText("##ren", g_rename.buffer, sizeof(g_rename.buffer),
                                       ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        bool lost  = ImGui::IsItemDeactivated();
        if (enter || lost) {
            std::string nn(g_rename.buffer);
            if (!nn.empty() && nn != node.name) { se.onRenameNode(node.name, nn); nodeModified = true; }
            g_rename.nodeName.clear();
        }
    } else {
        // ── Normal selectable ────────────────────────────────────────────
        if (inSel) {
            ImVec4 tint = isPrimary ? ImVec4{0.26f,0.59f,0.98f,0.50f}
                                    : ImVec4{0.26f,0.59f,0.98f,0.22f};
            ImGui::PushStyleColor(ImGuiCol_Header,        tint);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {tint.x,tint.y,tint.z, tint.w+0.15f});
        }
        clicked = ImGui::Selectable(node.name.c_str(), inSel,
                                     ImGuiSelectableFlags_SpanAllColumns);
        if (inSel) ImGui::PopStyleColor(2);

        // ── Drag & Drop (reorder) ────────────────────────────────────────
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
            HierarchyDragPayload pl{};
            snprintf(pl.nodeName, sizeof(pl.nodeName), "%s", node.name.c_str());
            snprintf(pl.parentGroup, sizeof(pl.parentGroup), "%s", parentGroup.c_str());
            ImGui::SetDragDropPayload("HIER_NODE", &pl, sizeof(pl));
            ImGui::TextColored(badgeCol, "%s", s.badge);
            ImGui::SameLine(0, 6);
            ImGui::TextUnformatted(node.name.c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("HIER_NODE")) {
                auto* data = (const HierarchyDragPayload*)p->Data;
                if (std::string(data->parentGroup) == parentGroup) {
                    bool after = ImGui::GetMousePos().y > (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) * 0.5f;
                    se.onReorderNode(data->nodeName, node.name, parentGroup, after);
                }
            }
            ImGui::EndDragDropTarget();
        }
    }

    // Lock indicator
    if (locked && g_rename.nodeName != node.name) {
        ImGui::SameLine(0, 4);
        ImGui::TextColored({1.0f, 0.5f, 0.0f, 0.7f}, "[L]");
    }

    // Right-click context menu
    if (!nodeModified && ImGui::BeginPopupContextItem("##nodectx")) {
        if (isTopLevel) {
            bool lk = se.isNodeLocked(node.name);
            if (ImGui::MenuItem(lk ? "Unlock" : "Lock"))
                se.onToggleLock(node.name);
        }
        if (ImGui::MenuItem("Rename")) {
            g_rename.nodeName = node.name;
            snprintf(g_rename.buffer, sizeof(g_rename.buffer), "%s", node.name.c_str());
            g_rename.focusSet = false;
        }
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 0.3f, 0.3f, 1.0f});
        if (ImGui::MenuItem("Delete")) { se.onDeleteNode(node.name); nodeModified = true; }
        ImGui::PopStyleColor();
        ImGui::EndPopup();
    }

    if (!nodeModified && clicked)
        se.onToggleSelection(node.ptr, node.name, ctrlHeld);

    ImGui::PopID();
}

// ─────────────────────────────────────────────────────────────────────────────
// Main menu bar
// ─────────────────────────────────────────────────────────────────────────────
inline void drawMenuBar(sceneEditor& se) {
    if (!ImGui::BeginMainMenuBar()) return;

    auto* gfx = se.baseGraphics();

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Save Scene", "Ctrl+S"))
            se.saveSceneToJson(std::filesystem::path(se.state().sceneFilePath));
        if (ImGui::MenuItem("Save Scene As..."))
            se.state().showSaveAsDialog = true;
        ImGui::Separator();
        moon::tests::gui::makeScreenshot("Screenshot", se.getApp());
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Esc")) se.getWindow().close();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Add")) {
        if (ImGui::MenuItem("Object..."))
            se.state().showAddObjectDialog = true;
        if (ImGui::BeginMenu("Light")) {
            if (ImGui::MenuItem("Point Light"))       se.onSpawnPointLight();
            if (ImGui::MenuItem("Spot Light"))        se.onSpawnSpotLight();
            if (ImGui::MenuItem("Directional Light")) se.onSpawnDirectionalLight();
            if (ImGui::MenuItem("Isotropic Light"))   se.onSpawnIsotropicLight();
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Camera")) se.onSpawnCamera();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        auto& st = se.state();
        ImGui::MenuItem("Hierarchy",   nullptr, &st.showHierarchy);
        ImGui::MenuItem("Inspector",   nullptr, &st.showInspector);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Camera")) {
            if (auto* cam = dynamic_cast<moon::entities::BaseCamera*>(se.activeCamera())) {
                cam->translation() = moon::math::quat(0.0f, 0.0f, 0.0f, 0.0f);
                cam->rotation()    = moon::math::quat(1.0f, 0.0f, 0.0f, 0.0f);
                auto egIt = se.getEditorGroups().find(se.getActiveCameraName());
                if (egIt != se.getEditorGroups().end()) {
                    egIt->second.group->translation() = moon::math::quat(0.0f, 0.0f, 0.0f, 10.0f);
                    egIt->second.group->rotation()    = moon::math::quat(1.0f, 0.0f, 0.0f, 0.0f);
                    egIt->second.group->update();
                } else {
                    cam->update();
                }
            }
        }
        ImGui::EndMenu();
    }

    if (gfx && ImGui::BeginMenu("Debug")) {
        auto bbParam = moon::deferredGraphics::Names::BoundingBox::param;
        auto bbName  = moon::deferredGraphics::Names::BoundingBox::name;
        bool bbOn = gfx->getEnable(bbParam);
        if (ImGui::MenuItem("Bounding Boxes", nullptr, bbOn)) {
            gfx->setEnable(bbParam, !bbOn);
            gfx->requestUpdate(bbName);
        }
        ImGui::EndMenu();
    }

    // Right-aligned FPS readout
    {
        float fps = se.fps();
        char buf[40];
        snprintf(buf, sizeof(buf), "%.0f FPS  %.2f ms ", fps, fps > 0.0f ? 1000.0f / fps : 0.0f);
        float textW = ImGui::CalcTextSize(buf).x;
        ImGui::SetCursorPosX(ImGui::GetContentRegionMax().x - textW);
        ImVec4 col = fps >= 60.0f ? ImVec4(0.3f,1.0f,0.3f,1.0f) : ImVec4(1.0f,0.5f,0.2f,1.0f);
        ImGui::TextColored(col, "%s", buf);
    }

    ImGui::EndMainMenuBar();
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene Hierarchy panel (left side, fixed)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawHierarchy(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showHierarchy) return;

    const float menuH  = ImGui::GetFrameHeight();
    const float statH  = ImGui::GetFrameHeight() + 6.0f;
    const float dispH  = ImGui::GetIO().DisplaySize.y;
    const float panW   = 300.0f;
    const float availH = dispH - menuH - statH;
    const float hierH  = availH * 0.5f;

    // ── Top half: Scene Hierarchy ─────────────────────────────────────────────
    ImGui::SetNextWindowPos({ 0.0f, menuH }, ImGuiCond_Always);
    ImGui::SetNextWindowSize({ panW, hierH }, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;
    if (!ImGui::Begin("Scene Hierarchy", &st.showHierarchy, wf)) { ImGui::End(); return; }

    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##filter", "Search...", st.hierarchyFilter, sizeof(st.hierarchyFilter));
    ImGui::Separator();

    const char* flt = st.hierarchyFilter;

    const auto roots = se.getHierarchyRoots();
    for (const auto& node : roots)
        drawHierarchyNode(se, node, flt, true, "");

    ImGui::Separator();

    if (!st.selection.empty()) {
        ImGui::TextDisabled("%zu selected", st.selection.size());
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear")) se.onClearSelection();
    }

    if (st.selection.size() >= 2) {
        if (ImGui::Button("Group Selected", {-1.0f, 0.0f}))
            se.onGroupSelected();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Ctrl+click to multi-select, then group %zu items.",
                              st.selection.size());
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Properties panel (left side, bottom half — shows selected object/group)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawProperties(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showHierarchy) return;

    const float menuH  = ImGui::GetFrameHeight();
    const float statH  = ImGui::GetFrameHeight() + 6.0f;
    const float dispH  = ImGui::GetIO().DisplaySize.y;
    const float panW   = 300.0f;
    const float availH = dispH - menuH - statH;
    const float hierH  = availH * 0.5f;
    const float propH  = availH - hierH;

    ImGui::SetNextWindowPos({ 0.0f, menuH + hierH }, ImGuiCond_Always);
    ImGui::SetNextWindowSize({ panW, propH }, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;
    if (!ImGui::Begin("Properties", nullptr, wf)) { ImGui::End(); return; }

    const float w = panW - 20.0f;
    const float sliderW = w * 0.55f;   // single-row sliders: leave room for labels
    auto* cam = dynamic_cast<moon::entities::BaseCamera*>(se.activeCamera());

    // ── Find if a group is selected ──────────────────────────────────────────
    moon::transformational::Group* selGroup = nullptr;
    std::string selGroupName;
    for (const auto& [gname, eg] : se.getEditorGroups()) {
        if (st.selection.count(eg.group)) {
            selGroup     = eg.group;
            selGroupName = gname;
            break;
        }
    }

    if (!st.selectedObject && !selGroup) {
        ImGui::TextDisabled("No object selected.");
        ImGui::TextDisabled("Double-click in viewport or");
        ImGui::TextDisabled("click in hierarchy to select.");
    } else if (selGroup && !st.selectedObject) {
        // ── Group selected ───────────────────────────────────────────────────
        bool isCamGroup  = se.isCameraGroup(selGroupName);
        bool isLgtGroup  = se.isLightGroup(selGroupName);

        auto reqLight = [&se]() {
            if (auto* g = se.baseGraphics()) {
                g->requestUpdate(moon::deferredGraphics::Names::Shadow::name);
                g->requestUpdate(moon::deferredGraphics::Names::Scattering::name);
                g->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
            }
        };

        if (isCamGroup) {
            bool isActive = (selGroupName == se.getActiveCameraName());
            ImGui::TextColored({1.0f,0.3f,0.3f,1.0f}, "Camera: %s%s",
                               selGroupName.c_str(), isActive ? "  (active)" : "");
            if (!isActive) {
                if (ImGui::Button("Switch to Camera", {-1.0f, 0.0f}))
                    se.onSwitchToCamera(selGroupName);
            }
        } else if (isLgtGroup) {
            ImGui::TextColored({1.0f,0.85f,0.1f,1.0f}, "Light: %s", selGroupName.c_str());
        } else {
            ImGui::TextColored({0.9f,0.6f,0.2f,1.0f}, "Group: %s", selGroupName.c_str());
        }
        if (ImGui::Button("Focus (F)", {-1.0f, 0.0f})) se.onFocusSelected();
        bool grpLocked = se.isNodeLocked(selGroupName);

        // ── Transform ────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (grpLocked) ImGui::BeginDisabled();
            float fw = (w - 12.0f) / 3.0f;
            ImGui::Text("Translation");
            moon::tests::gui::transManipulator<0>(*selGroup, "X##grtr", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::transManipulator<1>(*selGroup, "Y##grtr", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::transManipulator<2>(*selGroup, "Z##grtr", fw);
            ImGui::Text("Rotation");
            moon::tests::gui::rotationmManipulator(*selGroup, cam);
            ImGui::SameLine(0, 10);
            moon::tests::gui::printQuaternion(selGroup->rotation());
            if (grpLocked) ImGui::EndDisabled();
        }

        // ── Camera properties ────────────────────────────────────────────────
        if (isCamGroup) {
            auto camIt = se.getCameras().find(selGroupName);
            auto* selCam = (camIt != se.getCameras().end())
                ? dynamic_cast<moon::entities::BaseCamera*>(camIt->second.get()) : nullptr;
            if (selCam && ImGui::CollapsingHeader("Camera Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                static float fovDeg = 45.0f;
                ImGui::SetNextItemWidth(sliderW);
                if (ImGui::SliderFloat("FOV##camprop", &fovDeg, 10.0f, 120.0f)) {
                    selCam->setProjMatrix(moon::math::perspective(
                        moon::math::radians(fovDeg), se.getWindow().aspectRatio(), 0.1f));
                }
                auto camWorldPos = (selCam->globalTransformation() *
                    moon::math::vec4(selCam->translation().im()[0], selCam->translation().im()[1],
                                     selCam->translation().im()[2], 1.0f)).dvec();
                ImGui::Text("Pos  X: %.2f  Y: %.2f  Z: %.2f", camWorldPos[0], camWorldPos[1], camWorldPos[2]);
                if (ImGui::Button("Reset Rotation##camprop", {-1.0f, 0.0f})) {
                    selCam->rotation() = moon::math::quat(1.0f, 0.0f, 0.0f, 0.0f);
                    selCam->update();
                }
            }
        }

        // ── Light properties ─────────────────────────────────────────────────
        if (isLgtGroup) {
            auto egIt = se.getEditorGroups().find(selGroupName);
            auto* lptr = (egIt != se.getEditorGroups().end()) ? egIt->second.lightEntity : nullptr;
            if (lptr && ImGui::CollapsingHeader("Light Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (auto* dl = dynamic_cast<moon::entities::DirectionalLight*>(lptr)) {
                    bool en = dl->isEnable();
                    if (ImGui::Checkbox("Enabled##lp", &en)) { dl->setEnable(en); reqLight(); }
                    if (moon::tests::gui::directionalLightSliders(*dl, sliderW)) reqLight();
                } else if (auto* sl = dynamic_cast<moon::entities::SpotLight*>(lptr)) {
                    bool slEn = sl->getPower() > 0.0f;
                    if (ImGui::Checkbox("Enabled##slp", &slEn)) {
                        se.setLightEnabled(selGroupName, slEn);
                        reqLight();
                    }
                    if (moon::tests::gui::spotLightSliders(*sl, 0, sliderW)) reqLight();
                    moon::tests::gui::spotLightProjectionSliders(*sl, 0, sliderW);
                } else if (auto* iso = dynamic_cast<moon::entities::IsotropicLight*>(lptr)) {
                    moon::math::vec4 color = iso->getColor();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::ColorEdit4("Color##iso", (float*)&color, ImGuiColorEditFlags_NoDragDrop))
                        { iso->setColor(color); reqLight(); }
                    float power = iso->getPower();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Power##iso", &power, 0.0f, 100.0f))
                        { iso->setPower(power); reqLight(); }
                    float drop = iso->getDrop();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Drop##iso", &drop, 0.0f, 1.0f))
                        { iso->setDrop(drop); reqLight(); }
                    float inner = iso->getInnerFraction();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Inner##iso", &inner, 0.0f, 1.0f))
                        { iso->setInnerFraction(inner); reqLight(); }
                    float exp = iso->getExponent();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Exponent##iso", &exp, 0.0f, 20.0f))
                        { iso->setExponent(exp); reqLight(); }
                    bool shadow = iso->getEnableShadow();
                    if (ImGui::Checkbox("Shadow##iso", &shadow))
                        { iso->setEnableShadow(shadow); reqLight(); }
                    ImGui::SameLine();
                    bool scattering = iso->getEnableScattering();
                    if (ImGui::Checkbox("Scattering##iso", &scattering))
                        { iso->setEnableScattering(scattering); reqLight(); }
                } else if (auto* pl = dynamic_cast<moon::entities::PointLight*>(lptr)) {
                    moon::math::vec4 color = pl->getColor();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::ColorEdit4("Color##pl", (float*)&color, ImGuiColorEditFlags_NoDragDrop))
                        { pl->setColor(color); reqLight(); }
                    float power = pl->getPower();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Power##pl", &power, 0.0f, 100.0f))
                        { pl->setPower(power); reqLight(); }
                    float drop = pl->getDrop();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Drop##pl", &drop, 0.0f, 1.0f))
                        { pl->setDrop(drop); reqLight(); }
                    float radius = pl->getRadius();
                    ImGui::SetNextItemWidth(sliderW);
                    if (ImGui::SliderFloat("Radius##pl", &radius, 0.1f, 100.0f))
                        { pl->setRadius(radius); reqLight(); }
                }
            }
        }
    } else {
        // ── Object selected ──────────────────────────────────────────────────
        ImGui::TextColored({0.4f,0.8f,1.0f,1.0f}, "Selected: %s", st.selectedName.c_str());

        if (ImGui::Button("Focus (F)", {-1.0f, 0.0f})) se.onFocusSelected();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move camera toward the selected object.");

        std::string groupKey;
        for (auto& [key, g] : se.getGroups()) {
            if (g->find(st.selectedObject)) { groupKey = key; break; }
        }

        bool objLocked = se.isNodeLocked(st.selectedName);

        // ── Transform ────────────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (objLocked) ImGui::BeginDisabled();
            float fw = (w - 12.0f) / 3.0f;
            ImGui::Text("Translation");
            moon::tests::gui::transManipulator<0>(*st.selectedObject, "X##tr", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::transManipulator<1>(*st.selectedObject, "Y##tr", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::transManipulator<2>(*st.selectedObject, "Z##tr", fw);

            ImGui::Text("Scale");
            moon::tests::gui::scaleManipulator<0>(*st.selectedObject, "X##sc", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::scaleManipulator<1>(*st.selectedObject, "Y##sc", fw);
            ImGui::SameLine(0,4);
            moon::tests::gui::scaleManipulator<2>(*st.selectedObject, "Z##sc", fw);
            if (ImGui::SmallButton("Max scale"))
                st.selectedObject->scale({ st.selectedObject->scaling().maxValue() });
            ImGui::SameLine();
            if (ImGui::SmallButton("Min scale"))
                st.selectedObject->scale({ st.selectedObject->scaling().minValue() });

            ImGui::Text("Rotation");
            moon::tests::gui::rotationmManipulator(*st.selectedObject, cam);
            ImGui::SameLine(0, 10);
            moon::tests::gui::printQuaternion(st.selectedObject->rotation());
            if (objLocked) ImGui::EndDisabled();
        }

        // ── Appearance ───────────────────────────────────────────────────────
        auto* baseObj = dynamic_cast<moon::entities::BaseObject*>(st.selectedObject);
        if (baseObj && ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Outline", &st.outlineEnabled))
                se.onSyncOutlines();
            if (st.outlineEnabled) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(sliderW - 60.0f);
                if (ImGui::ColorEdit4("##oc", (float*)&st.outlineColor,
                        ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_NoLabel))
                    se.onSyncOutlines();
            }
            moon::tests::gui::setColors(st.selectedObject, sliderW);
        }

        // ── Material (PLY only) ──────────────────────────────────────────────
        if (baseObj) {
            auto* iface = baseObj->object();
            auto* plyModel = iface ? dynamic_cast<moon::models::PlyModel*>(iface->model()) : nullptr;
            if (plyModel && ImGui::CollapsingHeader("Material")) {
                if (moon::tests::gui::setPlyMaterial(plyModel)) se.requestUpdate();
            }
        }

        // ── Animation ────────────────────────────────────────────────────────
        if (baseObj && baseObj->animation.count() > 0 &&
            ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SetNextItemWidth(sliderW);
            ImGui::SliderFloat("Speed##anim", &se.animSpeed(), 0.0f, 5.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global animation speed multiplier.");

            static const char* clips[] = { "stop","0","1","2","3","4","5","6","7","8","9" };
            int sel = baseObj->animation.current() + 1;
            ImGui::SetNextItemWidth(sliderW);
            if (ImGui::Combo("Clip##anim", &sel, clips,
                    std::min(IM_ARRAYSIZE(clips), (int)baseObj->animation.count() + 1))) {
                if (sel == 0) baseObj->animation.stop();
                else           baseObj->animation.play(sel - 1, 0.2f);
            }
        }

        // ── Morph Weights ────────────────────────────────────────────────────
        if (baseObj && baseObj->morphTargetCount() > 0 &&
            ImGui::CollapsingHeader("Morph Weights", ImGuiTreeNodeFlags_DefaultOpen))
        {
            auto names = baseObj->morphTargetNames();
            uint32_t count = baseObj->morphTargetCount();
            for (uint32_t i = 0; i < count; i++) {
                std::string label = (i < names.size() && !names[i].empty())
                    ? names[i] : ("Target " + std::to_string(i));
                label += "##morph" + std::to_string(i);
                float val = baseObj->getMorphWeight(i);
                ImGui::SetNextItemWidth(sliderW);
                if (ImGui::SliderFloat(label.c_str(), &val, 0.0f, 1.0f)) {
                    baseObj->setMorphWeight(i, val);
                    if (auto* g = se.baseGraphics()) {
                        g->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
                        g->requestUpdate(moon::deferredGraphics::Names::Shadow::name);
                    }
                }
            }
        }

        // ── Attached Spot Lights ─────────────────────────────────────────────
        if (!groupKey.empty()) {
            auto reqLight = [&se]() {
                if (auto* g = se.baseGraphics()) {
                    g->requestUpdate(moon::deferredGraphics::Names::Shadow::name);
                    g->requestUpdate(moon::deferredGraphics::Names::Scattering::name);
                    g->requestUpdate(moon::deferredGraphics::Names::MainGraphics::name);
                }
            };
            auto it = se.getGroupLights().find(groupKey);
            if (it != se.getGroupLights().end() && !it->second.empty() &&
                ImGui::CollapsingHeader("Spot Lights"))
            {
                for (int i = 0; i < (int)it->second.size(); ++i) {
                    ImGui::PushID(i);
                    ImGui::Text("Light %d", i);
                    if (moon::tests::gui::spotLightSliders(*it->second[i], i, sliderW)) reqLight();
                    moon::tests::gui::spotLightProjectionSliders(*it->second[i], i, sliderW);
                    if (i + 1 < (int)it->second.size()) ImGui::Separator();
                    ImGui::PopID();
                }
            }
        }
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspector panel (right side, fixed)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawInspector(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showInspector) return;

    const float menuH  = ImGui::GetFrameHeight();
    const float statH  = ImGui::GetFrameHeight() + 6.0f;
    const float dispW  = ImGui::GetIO().DisplaySize.x;
    const float dispH  = ImGui::GetIO().DisplaySize.y;
    const float panW   = 360.0f;
    const float perfH  = 130.0f;  // fixed height for performance block
    const float totalH = dispH - menuH - statH;

    // ── Upper part: Post-Processing (scrollable) ────────────────────────────
    ImGui::SetNextWindowPos({ dispW - panW, menuH }, ImGuiCond_Always);
    ImGui::SetNextWindowSize({ panW, totalH - perfH }, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;
    if (!ImGui::Begin("Inspector", &st.showInspector, wf)) { ImGui::End(); return; }

    const float w = panW - 20.0f;
    const float sliderW = w * 0.55f;

    // ── Render Settings ─────────────────────────────────────────────────────
    if (auto* gfx = se.baseGraphics()) {
        if (ImGui::CollapsingHeader("Render Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("sec_render");

            // Resolution scale
            float& resScale = se.state().resolutionScale;
            auto curExt = gfx->parameters().extent;
            ImGui::SetNextItemWidth(sliderW);
            ImGui::SliderFloat("Resolution Scale", &resScale, 0.1f, 1.0f, "%.2f");
            ImGui::SameLine();
            if (ImGui::SmallButton("Apply##res"))
                se.state().pendingResolutionChange = true;
            ImGui::TextDisabled("Current: %ux%u", curExt[0], curExt[1]);

            // Viewport position in window
            ImGui::Separator();
            ImGui::Text("Viewport Position");
            auto pos = gfx->getPositionInWindow();
            bool posChanged = false;
            ImGui::SetNextItemWidth(sliderW);
            posChanged |= ImGui::SliderFloat("Offset X", &pos.offset[0], 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(sliderW);
            posChanged |= ImGui::SliderFloat("Offset Y", &pos.offset[1], 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(sliderW);
            posChanged |= ImGui::SliderFloat("Size X",   &pos.size[0],   0.01f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(sliderW);
            posChanged |= ImGui::SliderFloat("Size Y",   &pos.size[1],   0.01f, 1.0f, "%.2f");
            if (posChanged)
                gfx->setPositionInWindow(pos);

            if (ImGui::SmallButton("Reset All##viewport")) {
                resScale = 1.0f;
                se.state().pendingResolutionChange = true;
                gfx->setPositionInWindow({ {0.0f, 0.0f}, {1.0f, 1.0f} });
            }

            ImGui::PopID();
        }
    }

    // ── Post-Processing ───────────────────────────────────────────────────────
    if (auto* gfx = se.baseGraphics()) {
        auto sp = noOwn(gfx);
        using N = moon::deferredGraphics::Names;

        // -- SSAO --
        if (ImGui::CollapsingHeader("SSAO", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("sec_ssao");
            moon::tests::gui::switcher(sp, N::SSAO::param, N::SSAO::name);
            moon::tests::gui::ssaoProps(sp, sliderW);
            ImGui::PopID();
        }
        // -- Bloom --
        if (ImGui::CollapsingHeader("Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::PushID("sec_bloom");
            moon::tests::gui::switcher(sp, N::Bloom::param, N::Bloom::name);
            moon::tests::gui::bloomProps(sp, sliderW);
            ImGui::PopID();
        }
        // -- Blur --
        if (ImGui::CollapsingHeader("Blur")) {
            ImGui::PushID("sec_blur");
            moon::tests::gui::switcher(sp, N::Blur::param, N::Blur::name);
            moon::tests::gui::blurProps(sp, sliderW);
            ImGui::PopID();
        }
        // -- Scattering --
        if (ImGui::CollapsingHeader("Scattering")) {
            ImGui::PushID("sec_scatter");
            moon::tests::gui::switcher(sp, N::Scattering::param, N::Scattering::name);
            moon::tests::gui::scatteringProps(sp, sliderW);
            ImGui::PopID();
        }
        // -- Other toggles --
        if (ImGui::CollapsingHeader("Other")) {
            ImGui::PushID("sec_other");
            moon::tests::gui::switcher(sp, N::Skybox::param,      N::Skybox::name);
            moon::tests::gui::switcher(sp, N::SSLR::param,        N::SSLR::name);
            moon::tests::gui::switcher(sp, N::Shadow::param,      N::Shadow::name);
            moon::tests::gui::switcher(sp, N::BoundingBox::param, N::BoundingBox::name);
            ImGui::PopID();
        }
    }

    ImGui::End();

    // ── Lower part: Performance (fixed, no gap) ─────────────────────────────
    ImGui::SetNextWindowPos({ dispW - panW, menuH + totalH - perfH }, ImGuiCond_Always);
    ImGui::SetNextWindowSize({ panW, perfH }, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 8.0f, 6.0f });

    ImGuiWindowFlags perfWf = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                              ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                              ImGuiWindowFlags_NoScrollbar;
    ImGui::Begin("Performance", nullptr, perfWf);
    ImGui::PopStyleVar();
    {
        static constexpr int kHistorySize = 120;
        static constexpr float kSmoothing  = 0.05f;
        static float smoothFps = 0.0f;
        static float history[kHistorySize] = {};
        static int   histIdx = 0;

        float rawFps = se.fps();
        smoothFps = (smoothFps < 1.0f) ? rawFps : smoothFps + kSmoothing * (rawFps - smoothFps);

        history[histIdx] = smoothFps;
        histIdx = (histIdx + 1) % kHistorySize;

        float hMin = 1e9f, hMax = 0.0f, hAvg = 0.0f;
        for (int i = 0; i < kHistorySize; i++) {
            float v = history[i];
            if (v > 0.0f) { hMin = std::min(hMin, v); hMax = std::max(hMax, v); }
            hAvg += v;
        }
        hAvg /= kHistorySize;
        if (hMin > 1e8f) hMin = 0.0f;

        ImVec4 col = smoothFps >= 60.0f ? ImVec4(0.3f,1.0f,0.3f,1.0f)
                                        : ImVec4(1.0f,0.5f,0.2f,1.0f);
        ImGui::TextColored(col, "FPS: %.0f  |  %.2f ms", smoothFps,
                           smoothFps > 0.0f ? 1000.0f / smoothFps : 0.0f);

        float ordered[kHistorySize];
        for (int i = 0; i < kHistorySize; i++)
            ordered[i] = history[(histIdx + i) % kHistorySize];

        char overlay[64];
        snprintf(overlay, sizeof(overlay), "avg %.0f", hAvg);
        float plotH = ImGui::GetContentRegionAvail().y - 4.0f;
        if (plotH < 30.0f) plotH = 30.0f;
        ImGui::PlotLines("##fpsPlot", ordered, kHistorySize, 0, overlay,
                         hMin * 0.9f, hMax * 1.1f, { w, plotH });
    }
    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance overlay (semi-transparent, near top-left of viewport)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawPerformance(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showPerformance) return;

    const float menuH = ImGui::GetFrameHeight();
    ImGui::SetNextWindowPos({ 308.0f, menuH + 8.0f }, ImGuiCond_Once);
    ImGui::SetNextWindowSize({ 280.0f, 0.0f }, ImGuiCond_Once);
    ImGui::SetNextWindowBgAlpha(0.72f);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration |
                          ImGuiWindowFlags_NoBringToFrontOnFocus |
                          ImGuiWindowFlags_NoFocusOnAppearing |
                          ImGuiWindowFlags_NoNav |
                          ImGuiWindowFlags_NoScrollbar;
    ImGui::Begin("##perf", &st.showPerformance, wf);

    float fps = se.fps();
    ImVec4 col = fps >= 60.0f ? ImVec4(0.3f,1.0f,0.3f,1.0f) : ImVec4(1.0f,0.5f,0.2f,1.0f);
    ImGui::TextColored(col, "FPS: %.0f  |  %.2f ms", fps, fps > 0.0f ? 1000.0f/fps : 0.0f);
    moon::tests::gui::fpsPlot(fps, 120);

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Status bar (fixed strip at the very bottom)
// ─────────────────────────────────────────────────────────────────────────────
inline void drawStatusBar(sceneEditor& se) {
    const float dispW = ImGui::GetIO().DisplaySize.x;
    const float dispH = ImGui::GetIO().DisplaySize.y;
    const float barH  = ImGui::GetFrameHeight() + 6.0f;

    ImGui::SetNextWindowPos({ 0.0f, dispH - barH }, ImGuiCond_Always);
    ImGui::SetNextWindowSize({ dispW, barH }, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 8.0f, 3.0f });

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize  |
                          ImGuiWindowFlags_NoMove  | ImGuiWindowFlags_NoBringToFrontOnFocus |
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("##statusbar", nullptr, wf);
    ImGui::PopStyleVar();

    size_t objN    = se.getObjects().size();
    size_t lockedN = se.state().lockedNodes.size();
    size_t lightN  = se.getGroupLights().size() + (se.getSunLight() ? 1 : 0);
    float  fps     = se.fps();

    moon::math::vec3 camPos{0.0f};
    if (auto* cam = se.activeCamera()) {
        auto wp = cam->globalTransformation() *
            moon::math::vec4(cam->translation().im()[0], cam->translation().im()[1],
                             cam->translation().im()[2], 1.0f);
        camPos = wp.dvec();
    }

    const char* sel = se.state().selectedName.empty() ? "none" : se.state().selectedName.c_str();
    ImGui::Text("Objects: %zu  Locked: %zu  Lights: %zu  |  FPS: %.0f  |  "
                "Cam: (%.1f, %.1f, %.1f)  |  Sel: %s",
                objN, lockedN, lightN, fps, camPos[0], camPos[1], camPos[2], sel);

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Add Object dialog — scan a directory for supported model files
// ─────────────────────────────────────────────────────────────────────────────
struct DirNode {
    std::map<std::string, DirNode> subdirs;
    std::vector<std::pair<int, std::string>> files;   // (foundFiles index, filename)
};

inline void drawDirNode(const DirNode& node, int& selectedFileIdx) {
    for (auto& [name, child] : node.subdirs) {
        if (ImGui::TreeNodeEx(name.c_str(), ImGuiTreeNodeFlags_None)) {
            drawDirNode(child, selectedFileIdx);
            ImGui::TreePop();
        }
    }
    for (auto& [idx, fname] : node.files) {
        if (ImGui::Selectable(fname.c_str(), selectedFileIdx == idx))
            selectedFileIdx = idx;
    }
}

inline void drawAddObjectDialog(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showAddObjectDialog) return;

    // Persistent dialog state (survives across frames)
    static char dirPath[512] = {};
    static std::string lastScannedDir;
    static std::vector<std::filesystem::path> foundFiles;
    static DirNode dirTree;
    static int selectedFileIdx = -1;
    static float spawnScale = 1.0f;
    static bool animated = false;
    static bool firstOpen = true;

    // Set default path on first open
    if (firstOpen) {
        auto defaultDir = se.getExternalPath() / "dependences" / "model";
        auto u8str = defaultDir.u8string();
        std::string defaultStr(u8str.begin(), u8str.end());
        std::replace(defaultStr.begin(), defaultStr.end(), '/', '\\');
        snprintf(dirPath, sizeof(dirPath), "%s", defaultStr.c_str());
        firstOpen = false;
        lastScannedDir.clear(); // force rescan
    }

    ImGui::SetNextWindowSize(ImVec2(600, 450), ImGuiCond_Once);
    if (!ImGui::Begin("Add Object", &st.showAddObjectDialog)) {
        ImGui::End();
        return;
    }

    // Directory input
    ImGui::Text("Model directory:");
    bool pathChanged = ImGui::InputText("##dir", dirPath, sizeof(dirPath),
                                         ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::Button("Scan") || pathChanged || lastScannedDir.empty()) {
        std::string currentDir(dirPath);
        if (currentDir != lastScannedDir) {
            foundFiles.clear();
            dirTree = {};
            selectedFileIdx = -1;
            lastScannedDir = currentDir;

            std::error_code ec;
            if (std::filesystem::is_directory(currentDir, ec)) {
                for (auto& entry : std::filesystem::recursive_directory_iterator(currentDir, ec)) {
                    if (!entry.is_regular_file()) continue;
                    auto ext = entry.path().extension().u8string();
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c){ return std::tolower(c); });
                    if (ext == u8".glb" || ext == u8".gltf" || ext == u8".ply")
                        foundFiles.push_back(entry.path());
                }
                std::sort(foundFiles.begin(), foundFiles.end());

                // Build nested directory tree
                for (int i = 0; i < static_cast<int>(foundFiles.size()); ++i) {
                    std::error_code ec2;
                    auto rel = std::filesystem::relative(foundFiles[i], lastScannedDir, ec2);
                    if (ec2) continue;

                    DirNode* cur = &dirTree;
                    auto parent = rel.parent_path();
                    std::vector<std::string> parts;
                    for (auto& comp : parent) {
                        auto u8c = comp.u8string();
                        std::string s(u8c.begin(), u8c.end());
                        if (!s.empty() && s != ".") parts.push_back(s);
                    }
                    for (auto& p : parts) cur = &cur->subdirs[p];

                    auto fnU8 = rel.filename().u8string();
                    cur->files.push_back({ i, std::string(fnU8.begin(), fnU8.end()) });
                }
            }
        }
    }

    ImGui::Separator();

    // File list grouped by directory tree
    ImGui::Text("Found models: %d", static_cast<int>(foundFiles.size()));
    ImGui::BeginChild("##filelist", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 4), true);
    drawDirNode(dirTree, selectedFileIdx);
    ImGui::EndChild();

    ImGui::Separator();

    // Options
    ImGui::SliderFloat("Scale", &spawnScale, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Animated", &animated);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Enable for models with animations.\n"
                          "Allocates multiple resource buffers to\n"
                          "prevent conflicts during animation updates.");

    // Create button
    bool canCreate = (selectedFileIdx >= 0 && selectedFileIdx < static_cast<int>(foundFiles.size()));
    if (!canCreate) ImGui::BeginDisabled();
    if (ImGui::Button("Create", ImVec2(120, 0))) {
        st.pendingSpawn = { foundFiles[selectedFileIdx], spawnScale, animated, true };
        st.showAddObjectDialog = false;
    }
    if (!canCreate) ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0)))
        st.showAddObjectDialog = false;

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Save Scene As dialog
// ─────────────────────────────────────────────────────────────────────────────
inline void drawSaveAsDialog(sceneEditor& se) {
    auto& st = se.state();
    if (!st.showSaveAsDialog) return;

    ImGui::SetNextWindowSize(ImVec2(500, 0), ImGuiCond_Once);
    if (!ImGui::Begin("Save Scene As", &st.showSaveAsDialog)) {
        ImGui::End();
        return;
    }

    ImGui::Text("File path:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##savePath", st.sceneFilePath, sizeof(st.sceneFilePath));

    if (ImGui::Button("Save", ImVec2(120, 0))) {
        se.saveSceneToJson(std::filesystem::path(st.sceneFilePath));
        st.showSaveAsDialog = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0)))
        st.showSaveAsDialog = false;

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Master call: draw all editor panels from within an active ImGui frame
// ─────────────────────────────────────────────────────────────────────────────
inline void drawAll(sceneEditor& se) {
    drawMenuBar(se);
    drawAddObjectDialog(se);
    drawSaveAsDialog(se);
    drawHierarchy(se);
    drawProperties(se);
    drawInspector(se);
    drawStatusBar(se);
}

} // namespace editor::gui

#endif // EDITOR_GUI_H
