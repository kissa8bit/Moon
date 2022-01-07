cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(SOURCES
    graphicsLinker.cpp
    graphicsManager.cpp
)

set(HEADERS
    graphicsInterface.h
    graphicsLinker.h
    graphicsManager.h
    linkable.h
)

add_library(graphicsManager STATIC ${SOURCES} ${HEADERS})
target_include_directories(graphicsManager PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(graphicsManager PUBLIC utils math)
