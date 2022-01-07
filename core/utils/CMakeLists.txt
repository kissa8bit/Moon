cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(INCLUDE_DIRS
    ${Vulkan_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}/vulkan
    ${STB_DIR}
)

include_directories(${INCLUDE_DIRS})

set(SOURCES
    attachments.cpp
    pipelineNode.cpp
    operations.cpp
    texture.cpp
    device.cpp
    vkdefault.cpp
    buffer.cpp
    swapChain.cpp
    depthMap.cpp
    cursor.cpp
    memory.cpp
    queueFamily.cpp
)

set(HEADERS
    attachments.h
    pipelineNode.h
    operations.h
    texture.h
    device.h
    vkdefault.h
    buffer.h
    swapChain.h
    depthMap.h
    cursor.h
    memory.h
    queueFamily.h
    window.h
    types.h
)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(utils STATIC ${SOURCES} ${HEADERS})
target_include_directories(utils PUBLIC ${INCLUDE_DIRS} ${ROOT_DIR})
target_link_libraries(utils PUBLIC ${LIBS_DIRS})
target_compile_definitions(utils PUBLIC USE_STB_IMAGE)