cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(HEADERS
    quaternion.h
    dualQuaternion.h
    vector.h
    matrix.h
    linearAlgebra.h
)

set(SOURCES
    quaternion.cpp
    dualQuaternion.cpp
    vector.cpp
    matrix.cpp
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(math STATIC ${SOURCES} ${HEADERS})
target_include_directories(math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ROOT_DIR})
