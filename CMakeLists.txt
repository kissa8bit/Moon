cmake_minimum_required(VERSION 3.20)

project(Moon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED)

option(MOON_SHADER_EMBED "Embed compiled SPIR-V shaders into the binary (OFF = load from disk)" OFF)

include(cmake/CompileShaders.cmake)

set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/glfw)
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/stb)
set(GLTF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinygltf)
set(PLY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinyply)
set(OBJ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/tinyobj)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/imgui)
set(IMGUIZMO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependences/libs/imGuIZMO.quat)

if(MOON_SHADER_EMBED)
    add_compile_definitions(MOON_SHADER_EMBED)
else()
    # Expose build dir so runtime code can locate the generated SPV files
    add_compile_definitions(MOON_BUILD_PATH="${CMAKE_BINARY_DIR}")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(GLFW_LIB
        $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/Debug/glfw3dll.lib>
        $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3dll.lib>)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(GLFW_LIB ${GLFW_DIR}/build/src/libglfw.so.3.4)
endif()
set(LINK_LIBS ${Vulkan_LIBRARIES} ${GLFW_LIB})

message(STATUS "CMAKE_CURRENT_LIST_DIR is ${CMAKE_CURRENT_LIST_DIR}")
add_compile_definitions(ROOT_PATH="${CMAKE_CURRENT_LIST_DIR}")

add_subdirectory(core/utils)
add_subdirectory(core/workflows)
add_subdirectory(core/graphicsManager)
add_subdirectory(core/imguiGraphics)
add_subdirectory(core/interfaces)
add_subdirectory(core/implementations)
add_subdirectory(core/deferredGraphics)
add_subdirectory(core/models)
add_subdirectory(core/transformationals)
add_subdirectory(core/entities)
add_subdirectory(core/math)
add_subdirectory(tests/sceneEditor)

# Generate the combined shaders_embedded.h from all accumulated SPV files.
# Must come after all add_subdirectory calls that invoke moon_compile_shaders.
moon_generate_embed_header()
if(MOON_SHADER_EMBED)
    add_dependencies(utils moon_shaders_embedded)
endif()

# Convenience target: cmake --build . --target compile_shaders
add_custom_target(compile_shaders
    DEPENDS deferredGraphics_compile_shaders workflows_compile_shaders
)

option(CUDA_RAY_TRACING "Compile cudaRayTracing and cudaTest" OFF)
if(CUDA_RAY_TRACING)
    add_subdirectory(core/rayTracingGraphics/cudaRayTracing)
    add_subdirectory(core/rayTracingGraphics)
    add_subdirectory(tests/testCuda)
endif()
