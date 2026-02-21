cmake_minimum_required(VERSION 3.14)

set(SOURCES
    blur.cpp
    workflow.cpp
    postProcessing.cpp
    shadow.cpp
    skybox.cpp
    bloom.cpp
    ssao.cpp
    sslr.cpp
    scattering.cpp
    boundingBox.cpp
    selector.cpp
)

set(HEADERS
    blur.h
    workflow.h
    postProcessing.h
    shadow.h
    skybox.h
    bloom.h
    ssao.h
    sslr.h
    scattering.h
    boundingBox.h
    selector.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(workflows STATIC ${SOURCES} ${HEADERS})
target_include_directories(workflows PUBLIC ${ROOT_DIR})
target_link_libraries(workflows PUBLIC implementations interfaces utils)
