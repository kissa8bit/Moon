cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(renderStages)

set(SOURCES
    deferredGraphics.cpp
    renderStages/layersCombiner.cpp
    renderStages/source/ambientLighting.cpp
    renderStages/source/lighting.cpp
    renderStages/source/lightingPipelines.cpp
    renderStages/source/base.cpp
    renderStages/source/extension.cpp
    renderStages/graphics.cpp
    renderStages/deferredLink.cpp
)

set(HEADERS
    deferredGraphics.h
    renderStages/layersCombiner.h
    renderStages/graphics.h
    renderStages/deferredLink.h
    renderStages/deferredAttachments.h
    renderStages/layerIndex.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(RENDERSTAGES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/renderStages)

add_library(deferredGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(deferredGraphics PUBLIC ${ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${RENDERSTAGES_DIR})
target_link_libraries(deferredGraphics PUBLIC graphicsManager workflows implementations interfaces utils)
