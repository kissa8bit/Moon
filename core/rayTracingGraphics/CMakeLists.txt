cmake_minimum_required(VERSION 3.14)

project(rayTracingGraphics LANGUAGES CXX)

set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES
    rayTracingGraphics.cpp
    rayTracingLink.cpp
    boundingBoxGraphics.cpp
)

set(HEADERS
    rayTracingGraphics.h
    rayTracingLink.h
    boundingBoxGraphics.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(rayTracingGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(rayTracingGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
target_link_libraries(rayTracingGraphics PUBLIC graphicsManager cudaRayTracing workflows)
