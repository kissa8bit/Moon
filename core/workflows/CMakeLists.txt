cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

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

add_library(workflows STATIC ${SOURCES} ${HEADERS})
target_include_directories(workflows PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(workflows PUBLIC interfaces utils)
