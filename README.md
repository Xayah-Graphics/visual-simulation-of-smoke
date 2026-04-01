![banner](https://github.com/Xayah-Graphics/imagebed/blob/aa9d00c9c097a08e7883f1810dc32a4d3e959bb8/visual-simulation-of-smoke.png)
# [SIGGRAPH 2001] Visual Simulation of Smoke

[![Arch Build](https://github.com/Xayah-Graphics/visual-simulation-of-smoke/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/visual-simulation-of-smoke/actions/workflows/arch-build.yml)
[![Windows Build](https://github.com/Xayah-Graphics/visual-simulation-of-smoke/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/visual-simulation-of-smoke/actions/workflows/windows-build.yml)

Modern C++ 23 / CUDA Implementation with C ABI of the paper [_Visual simulation of smoke_](https://dl.acm.org/doi/10.1145/383259.383260) by Ronald Fedkiw et al.

## 1. Algorithm Pipeline

[TODO] On dev...

## 2. Build Instruction

#### Build C ABI library
- CMake 4.3.0 or higher
- Ninja build system (for CXX std module support)
- A C++23 compliant compiler (tested on Arch Linux with gcc/g++ 15.2.1, Windows with MSVC 17.14.29)
- NVIDIA CUDA 13.2 or higher

```
cmake -B build -S . -G Ninja
cmake --build build --parallel
```

#### Build with Vulkan visualizer

A built-in Vulkan visualizer is provided. To enable it, you need to install latest Vulkan SDK.

- Vulkan SDK 1.4 or higher.

```
cmake -B build -S . -G Ninja -DSTABLE_FLUIDS_BUILD_VULKAN_APP=ON
cmake --build build --parallel
```
