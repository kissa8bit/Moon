cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

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

add_library(transformationals STATIC ${SOURCES} ${HEADERS})
target_include_directories(transformationals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(transformationals PUBLIC math interfaces utils)
