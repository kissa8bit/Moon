cmake_minimum_required(VERSION 3.14)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLTF_DIR}
    ${PLY_DIR}/source
)

set(SOURCES
    gltfmodel/loadAnimations.cpp
    gltfmodel/loadMaterials.cpp
    gltfmodel/gltfmodel.cpp
    plymodel/plymodel.cpp
)

set(HEADERS
    gltfmodel/loadVertices.h
    gltfmodel/tinyGLTF.h
    gltfmodel/gltfutils.h
    gltfmodel/gltfmesh.h
    gltfmodel/skin.h
    gltfmodel/node.h
    gltfmodel/gltfskeleton.h
    gltfmodel.h
    plymodel.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(models STATIC ${SOURCES} ${HEADERS})
target_include_directories(models PUBLIC ${INCLUDE_DIRS} ${ROOT_DIR})
target_link_libraries(models PUBLIC interfaces utils)
