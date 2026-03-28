# cmake/CompileShaders.cmake
# Shader compilation helpers for the Moon engine.
#
# Requires:
#   cmake_minimum_required(VERSION 3.20)   — for DEPFILE support on all generators
#   find_package(Vulkan REQUIRED)          — provides Vulkan_GLSLC_EXECUTABLE
#   find_package(Python3 REQUIRED)         — needed when MOON_SHADER_EMBED=ON
#
# Options (set before including this file):
#   MOON_SHADER_EMBED  — OFF (default): load SPIR-V from disk at runtime
#                        ON:            embed SPIR-V as C arrays into the binary

# ---------------------------------------------------------------------------
# moon_compile_shaders(TARGET SHADER_SOURCE_DIR SPV_SUBDIR)
#
#   TARGET            — CMake target to attach shader compilation to
#   SHADER_SOURCE_DIR — root shader directory; immediate subdirs (except __*)
#                       are scanned for .vert / .frag / .comp files
#   SPV_SUBDIR        — output subdirectory under ${CMAKE_BINARY_DIR}/spv/
#                       e.g. "deferredGraphics" or "workflows"
#
# SPV files are written to:
#   ${CMAKE_BINARY_DIR}/spv/<SPV_SUBDIR>/<subdir>/<name><Vert|Frag|Comp>.spv
#
# When MOON_SHADER_EMBED=ON the SPV list is accumulated in a global property
# for moon_generate_embed_header() to consume later.
# ---------------------------------------------------------------------------
function(moon_compile_shaders TARGET SHADER_SOURCE_DIR SPV_SUBDIR)

    # Locate glslc (CMake 3.19+ sets Vulkan_GLSLC_EXECUTABLE via find_package)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        get_filename_component(_vk_bin "${Vulkan_INCLUDE_DIRS}/../../bin" ABSOLUTE)
        find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS "${_vk_bin}")
    endif()
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "[Moon] glslc not found. Is the Vulkan SDK installed?")
    endif()

    set(SPV_BASE "${CMAKE_BINARY_DIR}/spv/${SPV_SUBDIR}")
    set(ALL_SPV)

    file(GLOB _SUBDIRS LIST_DIRECTORIES true "${SHADER_SOURCE_DIR}/*")
    foreach(_SUBDIR_PATH ${_SUBDIRS})
        if(NOT IS_DIRECTORY "${_SUBDIR_PATH}")
            continue()
        endif()
        get_filename_component(_SUBDIR "${_SUBDIR_PATH}" NAME)
        if(_SUBDIR MATCHES "^__")
            continue()
        endif()

        file(GLOB _SHADERS
            "${_SUBDIR_PATH}/*.vert"
            "${_SUBDIR_PATH}/*.frag"
            "${_SUBDIR_PATH}/*.comp"
        )

        foreach(_SHADER ${_SHADERS})
            get_filename_component(_NAME_WE "${_SHADER}" NAME_WE)
            get_filename_component(_EXT     "${_SHADER}" EXT)

            if(_EXT STREQUAL ".vert")
                set(_SUFFIX "Vert")
            elseif(_EXT STREQUAL ".frag")
                set(_SUFFIX "Frag")
            elseif(_EXT STREQUAL ".comp")
                set(_SUFFIX "Comp")
            else()
                continue()
            endif()

            set(_SPV     "${SPV_BASE}/${_SUBDIR}/${_NAME_WE}${_SUFFIX}.spv")
            set(_DEPFILE "${SPV_BASE}/${_SUBDIR}/${_NAME_WE}${_SUFFIX}.d")

            add_custom_command(
                OUTPUT  "${_SPV}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPV_BASE}/${_SUBDIR}"
                COMMAND "${Vulkan_GLSLC_EXECUTABLE}" "${_SHADER}" -o "${_SPV}" -MD -MF "${_DEPFILE}"
                DEPENDS "${_SHADER}"
                DEPFILE "${_DEPFILE}"
                COMMENT "Compiling ${SPV_SUBDIR}/${_SUBDIR}/${_NAME_WE}${_SUFFIX}.spv"
            )
            list(APPEND ALL_SPV "${_SPV}")
        endforeach()
    endforeach()

    add_custom_target("${TARGET}_compile_shaders" DEPENDS ${ALL_SPV})
    add_dependencies("${TARGET}" "${TARGET}_compile_shaders")

    # Accumulate for embed header generation
    set_property(GLOBAL APPEND PROPERTY MOON_ALL_SPV_FILES ${ALL_SPV})
    set_property(GLOBAL APPEND PROPERTY MOON_ALL_COMPILE_TARGETS "${TARGET}_compile_shaders")

endfunction()

# ---------------------------------------------------------------------------
# moon_generate_embed_header()
#
# Call once from the root CMakeLists after all moon_compile_shaders calls.
# When MOON_SHADER_EMBED=ON:
#   - Generates ${CMAKE_BINARY_DIR}/generated/shaders_embedded.h at build time
#     (a combined header with all SPIR-V arrays + lookup function)
#   - Creates custom target  moon_shaders_embedded  that the utils library
#     must depend on so the header exists before C++ compilation.
# When MOON_SHADER_EMBED=OFF: no-op.
# ---------------------------------------------------------------------------
function(moon_generate_embed_header)
    if(NOT MOON_SHADER_EMBED)
        return()
    endif()

    get_property(ALL_SPV GLOBAL PROPERTY MOON_ALL_SPV_FILES)
    if(NOT ALL_SPV)
        message(WARNING "[Moon] moon_generate_embed_header: no SPV files accumulated.")
        return()
    endif()

    set(_HEADER "${CMAKE_BINARY_DIR}/generated/shaders_embedded.h")

    add_custom_command(
        OUTPUT  "${_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/generated"
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_SOURCE_DIR}/scripts/spv_to_header.py"
                "--output" "${_HEADER}"
                ${ALL_SPV}
        DEPENDS ${ALL_SPV}
        COMMENT "Generating shaders_embedded.h"
    )
    add_custom_target(moon_shaders_embedded ALL DEPENDS "${_HEADER}")

    # Ensure shader compilation runs before header generation.
    # add_custom_command DEPENDS cannot cross directory boundaries, so we
    # depend on the per-target compile targets instead.
    get_property(_COMPILE_TARGETS GLOBAL PROPERTY MOON_ALL_COMPILE_TARGETS)
    if(_COMPILE_TARGETS)
        add_dependencies(moon_shaders_embedded ${_COMPILE_TARGETS})
    endif()
endfunction()
