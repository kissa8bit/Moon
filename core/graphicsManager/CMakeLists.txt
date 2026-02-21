cmake_minimum_required(VERSION 3.14)

set(SOURCES
    graphicsLinker.cpp
    graphicsManager.cpp
)

set(HEADERS
    graphicsInterface.h
    graphicsLinker.h
    graphicsManager.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(graphicsManager STATIC ${SOURCES} ${HEADERS})
target_include_directories(graphicsManager PUBLIC ${ROOT_DIR})
target_link_libraries(graphicsManager PUBLIC utils math)
