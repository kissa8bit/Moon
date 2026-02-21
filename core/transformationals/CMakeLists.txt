cmake_minimum_required(VERSION 3.14)

set(SOURCES
    cameras.cpp
    objects.cpp
    lights.cpp
    group.cpp
)

set(HEADERS
    cameras.h
    objects.h
    lights.h
    transformational.h
    group.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(transformationals STATIC ${SOURCES} ${HEADERS})
target_include_directories(transformationals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
target_link_libraries(transformationals PUBLIC math implementations interfaces utils)
