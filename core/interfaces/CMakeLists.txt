cmake_minimum_required(VERSION 3.14)

set(SOURCES
    light.cpp
    model.cpp
    object.cpp
)

set(HEADERS
    camera.h
    light.h
    model.h
    object.h
    masksGeneration.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(interfaces STATIC ${SOURCES} ${HEADERS})
target_include_directories(interfaces PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
target_link_libraries(interfaces PUBLIC math utils)
