cmake_minimum_required(VERSION 3.5)

project(Moon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

find_package(Vulkan REQUIRED FATAL_ERROR)

set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/glfw)
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/stb)
set(GLTF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinygltf)
set(PLY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinyply)
set(OBJ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinyobj)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/imgui)
set(IMGUIZMO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/imGuIZMO.quat)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(GLFW_LIB
        $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/Debug/glfw3dll.lib>
        $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3dll.lib>)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(GLFW_LIB ${GLFW_DIR}/build/src/libglfw.so.3.4)
endif()
set(LIBS_DIRS ${Vulkan_LIBRARIES} ${GLFW_LIB})

add_subdirectory(core/utils)
add_subdirectory(core/workflows)
add_subdirectory(core/graphicsManager)
add_subdirectory(core/imguiGraphics)
add_subdirectory(core/interfaces)
add_subdirectory(core/deferredGraphics)
add_subdirectory(core/models)
add_subdirectory(core/transformationals)
add_subdirectory(core/math)
add_subdirectory(tests/testScene)

option(CUDA_RAY_TRACING "Compile cudaRayTracing and cudaTest" OFF)
if(CUDA_RAY_TRACING)
    add_subdirectory(core/rayTracingGraphics/cudaRayTracing)
    add_subdirectory(core/rayTracingGraphics)
    add_subdirectory(tests/testCuda)
endif()
