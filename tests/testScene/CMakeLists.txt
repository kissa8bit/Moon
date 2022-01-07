cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_DEBUG_POSTFIX d)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${GLFW_DIR}/include/GLFW
)

set(SOURCES
    testScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/gui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/glfwWindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/mouse.cpp
)
set(INCLUDES
    testScene.h
    controledObject.h
    testSceneGui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/scene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/controller.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/gui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/glfwWindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/mouse.h
)

add_executable(testScene ${INCLUDES} ${SOURCES})

target_link_libraries(testScene PRIVATE deferredGraphics imguiGraphics models entities transformationals ${LIBS_DIRS})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_custom_command(
        TARGET testScene POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:Debug>:${GLFW_DIR}/build/src/Debug/glfw3.dll>
        $<$<CONFIG:Release>:${GLFW_DIR}/build/src/Release/glfw3.dll>
        $<TARGET_FILE_DIR:testScene>
    )
endif()

