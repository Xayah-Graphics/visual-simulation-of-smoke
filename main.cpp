#include "visual-simulation-of-smoke.h"

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <numeric>
#include <vector>

int main() {
    try {
        nvtx3::scoped_range app_range{"vsmoke.demo"};
        auto cuda_ok = [](const cudaError_t status, const char* what) {
            if (status == cudaSuccess) {
                return true;
            }
            std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
            return false;
        };
        auto smoke_ok = [](const int32_t code, const char* what) {
            if (code == 0) {
                return true;
            }
            std::cerr << what << " failed (" << code << ")\n";
            return false;
        };

    constexpr int32_t nx = 48;
    constexpr int32_t ny = 72;
    constexpr int32_t nz = 48;
    constexpr float cell_size = 1.0f;
    constexpr float dt = 1.0f / 90.0f;
    constexpr float ambient_temperature = 0.0f;
    constexpr float density_buoyancy = 0.045f;
    constexpr float temperature_buoyancy = 0.12f;
    constexpr float vorticity_epsilon = 2.0f;
    constexpr int32_t pressure_iterations = 80;
    constexpr int32_t block_x = 8;
    constexpr int32_t block_y = 8;
    constexpr int32_t block_z = 4;
    constexpr uint32_t use_monotonic_cubic = 1u;

    const uint64_t scalar_bytes = visual_simulation_of_smoke_scalar_field_bytes(nx, ny, nz);
    const uint64_t velocity_x_bytes = visual_simulation_of_smoke_velocity_x_bytes(nx, ny, nz);
    const uint64_t velocity_y_bytes = visual_simulation_of_smoke_velocity_y_bytes(nx, ny, nz);
    const uint64_t velocity_z_bytes = visual_simulation_of_smoke_velocity_z_bytes(nx, ny, nz);
    const uint64_t workspace_bytes = visual_simulation_of_smoke_workspace_bytes(nx, ny, nz);

    float* density = nullptr;
    float* temperature = nullptr;
    float* velocity_x = nullptr;
    float* velocity_y = nullptr;
    float* velocity_z = nullptr;
    void* workspace = nullptr;
    cudaStream_t stream = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temperature), scalar_bytes), "cudaMalloc temperature") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z") ||
        !cuda_ok(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc workspace") ||
        !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        cudaFree(density);
        cudaFree(temperature);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(workspace);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        return EXIT_FAILURE;
    }

    if (!smoke_ok(visual_simulation_of_smoke_clear_async(density, scalar_bytes, temperature, scalar_bytes, velocity_x, velocity_x_bytes, velocity_y, velocity_y_bytes, velocity_z, velocity_z_bytes, nx, ny, nz, cell_size, stream), "visual_simulation_of_smoke_clear_async")) {
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 16; ++frame) {
        nvtx3::scoped_range frame_range{"vsmoke.demo.frame"};
        if (!smoke_ok(visual_simulation_of_smoke_add_source_async(density, scalar_bytes, temperature, scalar_bytes, velocity_x, velocity_x_bytes, velocity_y, velocity_y_bytes, velocity_z, velocity_z_bytes, nx, ny, nz, cell_size, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 0.85f, 1.35f, 0.0f, 1.2f, 0.0f, block_x, block_y, block_z, stream), "visual_simulation_of_smoke_add_source_async") ||
            !smoke_ok(visual_simulation_of_smoke_step_async(density, scalar_bytes, temperature, scalar_bytes, velocity_x, velocity_x_bytes, velocity_y, velocity_y_bytes, velocity_z, velocity_z_bytes, nx, ny, nz, cell_size, workspace, workspace_bytes, dt, ambient_temperature, density_buoyancy, temperature_buoyancy, vorticity_epsilon, pressure_iterations, block_x, block_y, block_z, use_monotonic_cubic, stream), "visual_simulation_of_smoke_step_async")) {
            return EXIT_FAILURE;
        }
    }

    if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        return EXIT_FAILURE;
    }

    std::vector<float> host_density(static_cast<size_t>(scalar_bytes / sizeof(float)), 0.0f);
    if (!cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) {
        return EXIT_FAILURE;
    }

    const float total_density = std::accumulate(host_density.begin(), host_density.end(), 0.0f);
    const float peak_density = host_density.empty() ? 0.0f : *std::max_element(host_density.begin(), host_density.end());

    std::cout << "visual-simulation-of-smoke-app\n";
    std::cout << "grid: " << nx << " x " << ny << " x " << nz << '\n';
    std::cout << "total density: " << total_density << '\n';
    std::cout << "peak density: " << peak_density << '\n';

    cudaStreamDestroy(stream);
    cudaFree(density);
    cudaFree(temperature);
    cudaFree(velocity_x);
    cudaFree(velocity_y);
    cudaFree(velocity_z);
    cudaFree(workspace);
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
