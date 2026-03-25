#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {} // namespace

int main() {
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
    constexpr int32_t frames               = 16;

    const uint64_t scalar_bytes     = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_x_bytes = static_cast<uint64_t>(nx + 1) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_y_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny + 1) * static_cast<uint64_t>(nz) * sizeof(float);
    const uint64_t velocity_z_bytes = static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny) * static_cast<uint64_t>(nz + 1) * sizeof(float);
    const std::size_t scalar_count  = static_cast<std::size_t>(scalar_bytes / sizeof(float));

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
    int exit_code                         = EXIT_SUCCESS;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temperature), scalar_bytes), "cudaMalloc temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_x), velocity_x_bytes), "cudaMalloc velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_y), velocity_y_bytes), "cudaMalloc velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&velocity_z), velocity_z_bytes), "cudaMalloc velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_density), scalar_bytes), "cudaMalloc temporary_previous_density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_temperature), scalar_bytes), "cudaMalloc temporary_previous_temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_x), velocity_x_bytes), "cudaMalloc temporary_previous_velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_y), velocity_y_bytes), "cudaMalloc temporary_previous_velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_previous_velocity_z), velocity_z_bytes), "cudaMalloc temporary_previous_velocity_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_pressure), scalar_bytes), "cudaMalloc temporary_pressure")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_divergence), scalar_bytes), "cudaMalloc temporary_divergence")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_x), scalar_bytes), "cudaMalloc temporary_omega_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_y), scalar_bytes), "cudaMalloc temporary_omega_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_z), scalar_bytes), "cudaMalloc temporary_omega_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_omega_magnitude), scalar_bytes), "cudaMalloc temporary_omega_magnitude")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_x), scalar_bytes), "cudaMalloc temporary_force_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_y), scalar_bytes), "cudaMalloc temporary_force_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&temporary_force_z), scalar_bytes), "cudaMalloc temporary_force_z")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) exit_code = EXIT_FAILURE;

    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(density, 0, scalar_bytes, stream), "cudaMemsetAsync density")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(temperature, 0, scalar_bytes, stream), "cudaMemsetAsync temperature")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_x, 0, velocity_x_bytes, stream), "cudaMemsetAsync velocity_x")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_y, 0, velocity_y_bytes, stream), "cudaMemsetAsync velocity_y")) exit_code = EXIT_FAILURE;
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemsetAsync(velocity_z, 0, velocity_z_bytes, stream), "cudaMemsetAsync velocity_z")) exit_code = EXIT_FAILURE;

    const auto cuda_begin = std::chrono::steady_clock::now();
    for (int frame = 0; exit_code == EXIT_SUCCESS && frame < frames; ++frame) {
        VisualSimulationOfSmokeAddScalarSourceDesc density_source_desc{
            .struct_size     = sizeof(VisualSimulationOfSmokeAddScalarSourceDesc),
            .api_version     = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx              = nx,
            .ny              = ny,
            .nz              = nz,
            .scalar          = density,
            .center_x        = static_cast<float>(nx) * 0.5f,
            .center_y        = static_cast<float>(ny) * 0.18f,
            .center_z        = static_cast<float>(nz) * 0.5f,
            .radius          = 4.5f,
            .amount          = 0.85f,
            .sample_offset_x = 0.5f,
            .sample_offset_y = 0.5f,
            .sample_offset_z = 0.5f,
            .block_x         = block_x,
            .block_y         = block_y,
            .block_z         = block_z,
            .stream          = stream,
        };

        VisualSimulationOfSmokeAddScalarSourceDesc temperature_source_desc{
            .struct_size     = sizeof(VisualSimulationOfSmokeAddScalarSourceDesc),
            .api_version     = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx              = nx,
            .ny              = ny,
            .nz              = nz,
            .scalar          = temperature,
            .center_x        = static_cast<float>(nx) * 0.5f,
            .center_y        = static_cast<float>(ny) * 0.18f,
            .center_z        = static_cast<float>(nz) * 0.5f,
            .radius          = 4.5f,
            .amount          = 1.35f,
            .sample_offset_x = 0.5f,
            .sample_offset_y = 0.5f,
            .sample_offset_z = 0.5f,
            .block_x         = block_x,
            .block_y         = block_y,
            .block_z         = block_z,
            .stream          = stream,
        };

        VisualSimulationOfSmokeAddVectorSourceDesc velocity_source_desc{
            .struct_size = sizeof(VisualSimulationOfSmokeAddVectorSourceDesc),
            .api_version = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx          = nx,
            .ny          = ny,
            .nz          = nz,
            .vector_x    = velocity_x,
            .vector_y    = velocity_y,
            .vector_z    = velocity_z,
            .center_x    = density_source_desc.center_x,
            .center_y    = density_source_desc.center_y,
            .center_z    = density_source_desc.center_z,
            .radius      = density_source_desc.radius,
            .amount_x    = 0.0f,
            .amount_y    = 1.2f,
            .amount_z    = 0.0f,
            .block_x     = block_x,
            .block_y     = block_y,
            .block_z     = block_z,
            .stream      = stream,
        };

        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_add_scalar_source_cuda(&density_source_desc), "visual_simulation_of_smoke_add_scalar_source_cuda(density)")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_add_scalar_source_cuda(&temperature_source_desc), "visual_simulation_of_smoke_add_scalar_source_cuda(temperature)")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_add_vector_source_cuda(&velocity_source_desc), "visual_simulation_of_smoke_add_vector_source_cuda")) exit_code = EXIT_FAILURE;

        VisualSimulationOfSmokeForcesDesc forces_desc{
            .struct_size               = sizeof(VisualSimulationOfSmokeForcesDesc),
            .api_version               = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx                        = nx,
            .ny                        = ny,
            .nz                        = nz,
            .cell_size                 = cell_size,
            .dt                        = dt,
            .ambient_temperature       = ambient_temperature,
            .density_buoyancy          = density_buoyancy,
            .temperature_buoyancy      = temperature_buoyancy,
            .vorticity_epsilon         = vorticity_epsilon,
            .density                   = density,
            .temperature               = temperature,
            .velocity_x                = velocity_x,
            .velocity_y                = velocity_y,
            .velocity_z                = velocity_z,
            .temporary_omega_x         = temporary_omega_x,
            .temporary_omega_y         = temporary_omega_y,
            .temporary_omega_z         = temporary_omega_z,
            .temporary_omega_magnitude = temporary_omega_magnitude,
            .temporary_force_x         = temporary_force_x,
            .temporary_force_y         = temporary_force_y,
            .temporary_force_z         = temporary_force_z,
            .block_x                   = block_x,
            .block_y                   = block_y,
            .block_z                   = block_z,
            .stream                    = stream,
        };

        VisualSimulationOfSmokeAdvectVelocityDesc advect_velocity_desc{
            .struct_size                   = sizeof(VisualSimulationOfSmokeAdvectVelocityDesc),
            .api_version                   = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx                            = nx,
            .ny                            = ny,
            .nz                            = nz,
            .cell_size                     = cell_size,
            .dt                            = dt,
            .use_monotonic_cubic           = use_monotonic_cubic,
            .velocity_x                    = velocity_x,
            .velocity_y                    = velocity_y,
            .velocity_z                    = velocity_z,
            .temporary_previous_velocity_x = temporary_previous_velocity_x,
            .temporary_previous_velocity_y = temporary_previous_velocity_y,
            .temporary_previous_velocity_z = temporary_previous_velocity_z,
            .block_x                       = block_x,
            .block_y                       = block_y,
            .block_z                       = block_z,
            .stream                        = stream,
        };

        VisualSimulationOfSmokeProjectDesc project_desc{
            .struct_size                   = sizeof(VisualSimulationOfSmokeProjectDesc),
            .api_version                   = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx                            = nx,
            .ny                            = ny,
            .nz                            = nz,
            .cell_size                     = cell_size,
            .dt                            = dt,
            .pressure_iterations           = pressure_iterations,
            .temporary_previous_velocity_x = temporary_previous_velocity_x,
            .temporary_previous_velocity_y = temporary_previous_velocity_y,
            .temporary_previous_velocity_z = temporary_previous_velocity_z,
            .temporary_pressure            = temporary_pressure,
            .temporary_divergence          = temporary_divergence,
            .temporary_omega_x             = temporary_omega_x,
            .temporary_omega_y             = temporary_omega_y,
            .block_x                       = block_x,
            .block_y                       = block_y,
            .block_z                       = block_z,
            .stream                        = stream,
        };

        VisualSimulationOfSmokeScalarFlowBinding scalar_bindings[2] = {
            VisualSimulationOfSmokeScalarFlowBinding{
                .scalar                    = density,
                .temporary_previous_scalar = temporary_previous_density,
                .clamp_non_negative        = 1u,
            },
            VisualSimulationOfSmokeScalarFlowBinding{
                .scalar                    = temperature,
                .temporary_previous_scalar = temporary_previous_temperature,
                .clamp_non_negative        = 0u,
            },
        };
        VisualSimulationOfSmokeAdvectScalarFlowDesc scalar_flow_desc{
            .struct_size         = sizeof(VisualSimulationOfSmokeAdvectScalarFlowDesc),
            .api_version         = VISUAL_SIMULATION_OF_SMOKE_API_VERSION,
            .nx                  = nx,
            .ny                  = ny,
            .nz                  = nz,
            .cell_size           = cell_size,
            .dt                  = dt,
            .use_monotonic_cubic = use_monotonic_cubic,
            .scalar_bindings     = scalar_bindings,
            .scalar_count        = 2,
            .velocity_x          = velocity_x,
            .velocity_y          = velocity_y,
            .velocity_z          = velocity_z,
            .block_x             = block_x,
            .block_y             = block_y,
            .block_z             = block_z,
            .stream              = stream,
        };

        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_forces_cuda(&forces_desc), "visual_simulation_of_smoke_forces_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_advect_velocity_cuda(&advect_velocity_desc), "visual_simulation_of_smoke_advect_velocity_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_project_cuda(&project_desc), "visual_simulation_of_smoke_project_cuda")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpyAsync(velocity_x, temporary_previous_velocity_x, velocity_x_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync velocity_x")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpyAsync(velocity_y, temporary_previous_velocity_y, velocity_y_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync velocity_y")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpyAsync(velocity_z, temporary_previous_velocity_z, velocity_z_bytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync velocity_z")) exit_code = EXIT_FAILURE;
        if (exit_code == EXIT_SUCCESS && !smoke_ok(visual_simulation_of_smoke_advect_scalar_flow_cuda(&scalar_flow_desc), "visual_simulation_of_smoke_advect_scalar_flow_cuda")) exit_code = EXIT_FAILURE;
    }
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) exit_code = EXIT_FAILURE;
    const auto cuda_end = std::chrono::steady_clock::now();

    std::vector<float> host_density(scalar_count, 0.0f);
    if (exit_code == EXIT_SUCCESS && !cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) exit_code = EXIT_FAILURE;

    const float cuda_total_density = exit_code == EXIT_SUCCESS ? std::accumulate(host_density.begin(), host_density.end(), 0.0f) : 0.0f;
    const float cuda_peak_density  = exit_code == EXIT_SUCCESS && !host_density.empty() ? *std::max_element(host_density.begin(), host_density.end()) : 0.0f;

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
    if (exit_code != EXIT_SUCCESS) return exit_code;

    const double cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_begin).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "visual-simulation-of-smoke benchmark\n";
    std::cout << "grid: " << nx << " x " << ny << " x " << nz << '\n';
    std::cout << "frames: " << frames << '\n';
    std::cout << "| metric | cuda |\n";
    std::cout << "|---|---:|\n";
    std::cout << "| total_ms | " << cuda_ms << " |\n";
    std::cout << "| step_ms | " << cuda_ms / static_cast<double>(frames) << " |\n";
    std::cout << "| total_density | " << cuda_total_density << " |\n";
    std::cout << "| peak_density | " << cuda_peak_density << " |\n";
    return EXIT_SUCCESS;
}
