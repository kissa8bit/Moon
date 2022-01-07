cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES
    testCuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/glfwWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/mouse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/gui.cpp
)

set(INCLUDES
    testCuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/scene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/controller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/glfwWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/mouse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/gui.h
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${OBJ_DIR}
    ${GLFW_DIR}/include/GLFW
)

add_executable(testCuda ${INCLUDES} ${SOURCES})
add_definitions(-DTESTCUDA)

target_link_libraries(testCuda PRIVATE rayTracingGraphics imguiGraphics utils ${LIBS_DIRS})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_custom_command(
        TARGET testCuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/Debug/glfw3.dll>
        $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3.dll>
        $<TARGET_FILE_DIR:testCuda>
    )
endif()

