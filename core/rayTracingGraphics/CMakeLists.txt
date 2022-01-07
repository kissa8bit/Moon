cmake_minimum_required(VERSION 3.5)

project(rayTracingGraphics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(rayTracingGraphics STATIC ${SOURCES} ${HEADERS})
target_include_directories(rayTracingGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rayTracingGraphics PUBLIC graphicsManager cudaRayTracing workflows)
