cmake_minimum_required(VERSION 3.14)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(INCLUDE_DIRS
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUIZMO_DIR}
    ${IMGUIZMO_DIR}/imGuIZMO.quat
)

set(SOURCES
    imguiGraphics.cpp
    imguiLink.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUIZMO_DIR}/imGuIZMO.quat/imGuIZMOquat.cpp
)

set(HEADERS
    imguiGraphics.h
    imguiLink.h
    ${IMGUI_DIR}/imgui.h
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.h
    ${IMGUI_DIR}/backends/imgui_impl_glfw.h
)

add_library(imguiGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(imguiGraphics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_DIR}/include
    ${ROOT_DIR}
)
target_compile_definitions(imguiGraphics PUBLIC IMGUIZMO_IMGUI_FOLDER=)
target_link_libraries(imguiGraphics PUBLIC graphicsManager utils)
