cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

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

add_library(implementations STATIC ${SOURCES} ${HEADERS})
target_include_directories(implementations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
target_link_libraries(implementations PUBLIC math interfaces utils)
