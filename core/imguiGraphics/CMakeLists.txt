cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(INCLUDE_DIRS
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMGUIZMO_DIR}
    ${IMGUIZMO_DIR}/imGuIZMO.quat
)

include_directories(
    ${INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_DIR}/include
)

add_compile_definitions(IMGUIZMO_IMGUI_FOLDER=)

set(SOURCES
    imguiGraphics.cpp
    imguiLink.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
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
    ${INCLUDE_DIRS}}
)
target_link_libraries(imguiGraphics PUBLIC graphicsManager utils)
