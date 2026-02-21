cmake_minimum_required(VERSION 3.14)

set(SOURCES
    spotLight.cpp
    baseCamera.cpp
    baseObject.cpp
    skyboxObject.cpp
)

set(HEADERS
    spotLight.h
    baseCamera.h
    baseObject.h
    skyboxObject.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(entities STATIC ${SOURCES} ${HEADERS})
target_include_directories(entities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
target_link_libraries(entities PUBLIC transformationals math implementations interfaces utils)
