cmake_minimum_required(VERSION 3.5)

project(cudaRayTracing LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_DEBUG_POSTFIX d)

set(SOURCES
    hitable/sphere.cu
    hitable/triangle.cu
    hitable/hitable.cu
    materials/material.cu
    models/model.cu
    models/objmodel.cu
    graphics/cudaRayTracing.cu
    transformational/camera.cu
    utils/hitableArray.cu
    utils/hitableContainer.cu
    utils/operations.cu
    utils/kdTree.cu
    utils/primitive.cu
    utils/texture.cu
)

set(HEADERS
    hitable/hitable.h
    hitable/sphere.h
    hitable/triangle.h
    materials/material.h
    math/quat2.h
    math/quat.h
    math/ray.h
    math/vec4.h
    math/box.h
    math/mat4.h
    models/model.h
    models/objmodel.h
    graphics/cudaRayTracing.h
    transformational/camera.h
    transformational/object.h
    utils/buffer.h
    utils/hitableArray.h
    utils/hitableContainer.h
    utils/timer.h
    utils/operations.h
    utils/devicep.h
    utils/kdTree.h
    utils/stack.h
    utils/primitive.h
    utils/texture.h
)

find_package(CUDA 12.0 REQUIRED)
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCH compute_86)
set(CUDA_SM sm_86)
set(SYSTEM_TYPE 64)
set(CUDA_NVCC_FLAGS
    ${CMAKE_CUDA_FLAGS}
    ${CUDA_NVCC_FLAGS}
    -gencode arch=${CUDA_ARCH},code=${CUDA_SM}
    --machine=${SYSTEM_TYPE}
    -rdc=true
    --use_fast_math
    --expt-extended-lambda
    --expt-relaxed-constexpr
    -std=c++17
)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OBJ_DIR}
    ${STB_DIR}
)

include_directories(${INCLUDE_DIRS})

add_library(cudaRayTracing STATIC ${SOURCES} ${HEADERS})
target_include_directories(cudaRayTracing PUBLIC ${INCLUDE_DIRS})
set_target_properties(cudaRayTracing
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       CUDA_STANDARD 17
       CXX_STANDARD 17
)
target_link_libraries(cudaRayTracing PUBLIC ${CUDA_LIBRARIES})
target_compile_options(cudaRayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS}>)
