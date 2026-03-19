#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

#include <nvtx3/nvtx3.hpp>

int main() {
    nvtx3::scoped_range app_range{"vsmoke.demo"};
    auto cuda_ok = [](const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
        return false;
    };
    auto smoke_ok = [](const int32_t code, const char* what) {
        if (code == 0) return true;
        std::cerr << what << " failed (" << code << ")\n";
        return false;
    };

    constexpr int32_t nx                   = 48;
    constexpr int32_t ny                   = 72;
    constexpr int32_t nz                   = 48;
    constexpr float cell_size              = 1.0f;
    constexpr float dt                     = 1.0f / 90.0f;
    constexpr float ambient_temperature    = 0.0f;
    constexpr float density_buoyancy       = 0.045f;
    constexpr float temperature_buoyancy   = 0.12f;
    constexpr float vorticity_epsilon      = 2.0f;
    constexpr int32_t pressure_iterations  = 80;
    constexpr int32_t block_x              = 8;
    constexpr int32_t block_y              = 8;
    constexpr int32_t block_z              = 4;
    constexpr uint32_t use_monotonic_cubic = 1u;

    const uint64_t scalar_bytes           = visual_simulation_of_smoke_scalar_field_bytes(nx, ny, nz);
    const uint64_t velocity_x_bytes       = visual_simulation_of_smoke_velocity_x_bytes(nx, ny, nz);
    const uint64_t velocity_y_bytes       = visual_simulation_of_smoke_velocity_y_bytes(nx, ny, nz);
    const uint64_t velocity_z_bytes       = visual_simulation_of_smoke_velocity_z_bytes(nx, ny, nz);
    float* density                        = nullptr;
    float* temperature                    = nullptr;
    float* velocity_x                     = nullptr;
    float* velocity_y                     = nullptr;
    float* velocity_z                     = nullptr;
    float* temporary_previous_density     = nullptr;
    float* temporary_previous_temperature = nullptr;
    float* temporary_previous_velocity_x  = nullptr;
    float* temporary_previous_velocity_y  = nullptr;
    float* temporary_previous_velocity_z  = nullptr;
    float* temporary_pressure             = nullptr;
    float* temporary_divergence           = nullptr;
    float* temporary_omega_x              = nullptr;
    float* temporary_omega_y              = nullptr;
    float* temporary_omega_z              = nullptr;
    float* temporary_omega_magnitude      = nullptr;
    float* temporary_force_x              = nullptr;
    float* temporary_force_y              = nullptr;
    float* temporary_force_z              = nullptr;
    cudaStream_t stream                   = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temperature), scalar_bytes), "cudaMalloc temperature") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_density), scalar_bytes), "cudaMalloc temporary_previous_density")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_temperature), scalar_bytes), "cudaMalloc temporary_previous_temperature") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_x), velocity_x_bytes), "cudaMalloc temporary_previous_velocity_x")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_y), velocity_y_bytes), "cudaMalloc temporary_previous_velocity_y") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_z), velocity_z_bytes), "cudaMalloc temporary_previous_velocity_z")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_pressure), scalar_bytes), "cudaMalloc temporary_pressure") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_divergence), scalar_bytes), "cudaMalloc temporary_divergence")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_x), scalar_bytes), "cudaMalloc temporary_omega_x") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_y), scalar_bytes), "cudaMalloc temporary_omega_y") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_z), scalar_bytes), "cudaMalloc temporary_omega_z")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_magnitude), scalar_bytes), "cudaMalloc temporary_omega_magnitude") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_x), scalar_bytes), "cudaMalloc temporary_force_x")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_y), scalar_bytes), "cudaMalloc temporary_force_y") || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_z), scalar_bytes), "cudaMalloc temporary_force_z") || !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        cudaFree(density);
        cudaFree(temperature);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(temporary_previous_density);
        cudaFree(temporary_previous_temperature);
        cudaFree(temporary_previous_velocity_x);
        cudaFree(temporary_previous_velocity_y);
        cudaFree(temporary_previous_velocity_z);
        cudaFree(temporary_pressure);
        cudaFree(temporary_divergence);
        cudaFree(temporary_omega_x);
        cudaFree(temporary_omega_y);
        cudaFree(temporary_omega_z);
        cudaFree(temporary_omega_magnitude);
        cudaFree(temporary_force_x);
        cudaFree(temporary_force_y);
        cudaFree(temporary_force_z);
        if (stream != nullptr) cudaStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    if (!smoke_ok(visual_simulation_of_smoke_clear_async(density, temperature, velocity_x, velocity_y, velocity_z, nx, ny, nz, stream), "visual_simulation_of_smoke_clear_async")) {
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 16; ++frame) {
        nvtx3::scoped_range frame_range{"vsmoke.demo.frame"};
        if (!smoke_ok(visual_simulation_of_smoke_add_source_async(density, temperature, velocity_x, velocity_y, velocity_z, nx, ny, nz, static_cast<float>(nx) * 0.5f, static_cast<float>(ny) * 0.18f, static_cast<float>(nz) * 0.5f, 4.5f, 0.85f, 1.35f, 0.0f, 1.2f, 0.0f, block_x, block_y, block_z, stream), "visual_simulation_of_smoke_add_source_async")
            || !smoke_ok(visual_simulation_of_smoke_step_async(density, temperature, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, temporary_previous_density, temporary_previous_temperature, temporary_previous_velocity_x, temporary_previous_velocity_y, temporary_previous_velocity_z, temporary_pressure, temporary_divergence, temporary_omega_x,
                             temporary_omega_y, temporary_omega_z, temporary_omega_magnitude, temporary_force_x, temporary_force_y, temporary_force_z, dt, ambient_temperature, density_buoyancy, temperature_buoyancy, vorticity_epsilon, pressure_iterations, block_x, block_y, block_z, use_monotonic_cubic, stream),
                "visual_simulation_of_smoke_step_async")) {
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
    const float peak_density  = host_density.empty() ? 0.0f : *std::max_element(host_density.begin(), host_density.end());

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
    cudaFree(temporary_previous_density);
    cudaFree(temporary_previous_temperature);
    cudaFree(temporary_previous_velocity_x);
    cudaFree(temporary_previous_velocity_y);
    cudaFree(temporary_previous_velocity_z);
    cudaFree(temporary_pressure);
    cudaFree(temporary_divergence);
    cudaFree(temporary_omega_x);
    cudaFree(temporary_omega_y);
    cudaFree(temporary_omega_z);
    cudaFree(temporary_omega_magnitude);
    cudaFree(temporary_force_x);
    cudaFree(temporary_force_y);
    cudaFree(temporary_force_z);
    return EXIT_SUCCESS;
}
