cmake_minimum_required(VERSION 3.14)

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
