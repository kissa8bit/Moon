cmake_minimum_required(VERSION 3.20)

project(cudaRayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_DEBUG_POSTFIX d)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    hitable/sphere.cu
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
    hitable/objectInstance.h
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

add_library(cudaRayTracing STATIC ${SOURCES} ${HEADERS})

target_include_directories(cudaRayTracing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OBJ_DIR}
    ${STB_DIR}
)

set_target_properties(cudaRayTracing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "120"
)

target_compile_options(cudaRayTracing PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-extended-lambda
        --expt-relaxed-constexpr
    >
)

target_link_libraries(cudaRayTracing PUBLIC CUDA::cudart)
