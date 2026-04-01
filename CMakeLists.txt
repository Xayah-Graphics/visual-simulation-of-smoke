cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(visual-simulation-of-smoke VERSION 1.0.0 LANGUAGES CXX CUDA)

option(SMOKE_BUILD_BENCHMARK "Build benchmark applications" OFF)
option(SMOKE_BUILD_VULKAN_APP "Build Vulkan sample application" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================================================================
# Libraries
# ============================================================================
find_package(CUDAToolkit REQUIRED)
add_library(visual-simulation-of-smoke SHARED)
target_sources(
        visual-simulation-of-smoke
        PRIVATE
        backend-cuda.cu
        PUBLIC
        visual-simulation-of-smoke-3d.h
)
target_link_libraries(visual-simulation-of-smoke PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
target_include_directories(visual-simulation-of-smoke PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(visual-simulation-of-smoke PRIVATE VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED)
target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
if (MSVC)
    target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
    target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
else ()
    target_compile_options(visual-simulation-of-smoke PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif ()


# ============================================================================
# Samples
# ============================================================================
if (SMOKE_BUILD_BENCHMARK)
    add_executable(visual-simulation-of-smoke-benchmark benchmark.cu)
    target_link_libraries(visual-simulation-of-smoke-benchmark PRIVATE visual-simulation-of-smoke CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_compile_options(visual-simulation-of-smoke-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    if (MSVC)
        target_compile_options(visual-simulation-of-smoke-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
        target_compile_options(visual-simulation-of-smoke-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    else ()
        target_compile_options(visual-simulation-of-smoke-benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
    endif ()
endif ()


# ============================================================================
# Vulkan Sample Application
# ============================================================================
if (SMOKE_BUILD_VULKAN_APP)
    find_package(Vulkan REQUIRED)
    set(_VULKAN_HPP_IMPL "${CMAKE_CURRENT_BINARY_DIR}/_vulkan_hpp_impl.cpp")
    file(WRITE "${_VULKAN_HPP_IMPL}" [[
#include <vulkan/vulkan_raii.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
]])
    if (NOT TARGET glfw)
        add_subdirectory(vulkan-app/vendor/glfw)
    endif ()
    # ============================================================================
    # Shader Compilation
    # ============================================================================
    find_program(
            SLANGC_EXECUTABLE
            NAMES slangc
            HINTS ENV VULKAN_SDK
            PATH_SUFFIXES Bin bin
    )
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(SMOKE_SIM_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(SMOKE_SIM_SHADER_SOURCES
            vulkan-app/shaders/background.slang
            vulkan-app/shaders/field_plane.slang
            vulkan-app/shaders/field_volume.slang
    )
    set(SMOKE_SIM_SHADER_OUTPUTS "")
    foreach (shader IN LISTS SMOKE_SIM_SHADER_SOURCES)
        get_filename_component(shader_name "${shader}" NAME_WE)
        set(shader_output "${SMOKE_SIM_SHADER_DIR}/${shader_name}.spv")
        add_custom_command(
                OUTPUT "${shader_output}"
                COMMAND "${SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" -profile sm_6_5 -target spirv -entry vs_main -entry fs_main -o "${shader_output}"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                COMMENT "Compiling ${shader_name}.slang"
                VERBATIM
        )
        list(APPEND SMOKE_SIM_SHADER_OUTPUTS "${shader_output}")
    endforeach ()
    add_custom_target(visual-simulation-of-smoke-shaders DEPENDS ${SMOKE_SIM_SHADER_OUTPUTS})
    # ============================================================================
    # Application
    # ============================================================================
    add_executable(visual-simulation-of-smoke-app)
    target_sources(visual-simulation-of-smoke-app
            PRIVATE
            vulkan-app/src/vk.camera.cpp
            vulkan-app/src/vk.context.cpp
            vulkan-app/src/vk.frame.cpp
            vulkan-app/src/vk.geometry.cpp
            vulkan-app/src/vk.imgui.cpp
            vulkan-app/src/vk.math.cpp
            vulkan-app/src/vk.memory.cpp
            vulkan-app/src/vk.pipeline.cpp
            vulkan-app/src/vk.swapchain.cpp
            vulkan-app/src/vk.texture.cpp
            vulkan-app/vendor/imgui/imgui.cpp
            vulkan-app/vendor/imgui/imgui_draw.cpp
            vulkan-app/vendor/imgui/imgui_tables.cpp
            vulkan-app/vendor/imgui/imgui_widgets.cpp
            vulkan-app/vendor/imgui/backends/imgui_impl_glfw.cpp
            vulkan-app/vendor/imgui/backends/imgui_impl_vulkan.cpp

            vulkan-app/main.cpp
            vulkan-app/app.cpp
            vulkan-app/scene_plume.cpp

            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
            vulkan-app/modules/vk.camera.ixx
            vulkan-app/modules/vk.context.ixx
            vulkan-app/modules/vk.frame.ixx
            vulkan-app/modules/vk.geometry.ixx
            vulkan-app/modules/vk.imgui.ixx
            vulkan-app/modules/vk.math.ixx
            vulkan-app/modules/vk.memory.ixx
            vulkan-app/modules/vk.pipeline.ixx
            vulkan-app/modules/vk.swapchain.ixx
            vulkan-app/modules/vk.texture.ixx
            vulkan-app/app.ixx
            vulkan-app/scene_plume.ixx
    )
    add_dependencies(visual-simulation-of-smoke-app visual-simulation-of-smoke-shaders)
    target_compile_definitions(visual-simulation-of-smoke-app PRIVATE SMOKE_SIM_SHADER_DIR="${SMOKE_SIM_SHADER_DIR}" VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
    target_link_libraries(visual-simulation-of-smoke-app PRIVATE Vulkan::Vulkan glfw visual-simulation-of-smoke CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_include_directories(visual-simulation-of-smoke-app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-app/vendor/imgui)
    set_property(TARGET visual-simulation-of-smoke-app PROPERTY CXX_MODULE_STD ON)
    set_property(TARGET visual-simulation-of-smoke-app PROPERTY CXX_SCAN_FOR_MODULES ON)
    set_target_properties(visual-simulation-of-smoke-app PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if (MSVC)
        target_compile_options(visual-simulation-of-smoke-app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:preprocessor>)
    endif ()
endif ()
