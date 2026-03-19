#include "visual-simulation-of-smoke.h"

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <numeric>
#include <vector>

namespace {

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

bool smoke_ok(int32_t code, const char* what, const VisualSimulationOfSmokeContext* context = nullptr) {
    if (code == VISUAL_SIMULATION_OF_SMOKE_SUCCESS) {
        return true;
    }

    const uint64_t message_length =
        context != nullptr ? visual_simulation_of_smoke_context_last_error_length(context) : visual_simulation_of_smoke_last_error_length();
    std::vector<char> message(static_cast<size_t>(message_length + 1), '\0');
    const int32_t copy_code = context != nullptr
        ? visual_simulation_of_smoke_copy_context_last_error(context, message.data(), static_cast<uint64_t>(message.size()))
        : visual_simulation_of_smoke_copy_last_error(message.data(), static_cast<uint64_t>(message.size()));

    std::cerr << what << " failed (" << code << ")";
    if (copy_code == VISUAL_SIMULATION_OF_SMOKE_SUCCESS && !message.empty() && message[0] != '\0') {
        std::cerr << ": " << message.data();
    }
    std::cerr << '\n';
    return false;
}

} // namespace

int main() {
    nvtx3::scoped_range app_range{"vsmoke.demo"};
    VisualSimulationOfSmokeContextDesc desc = visual_simulation_of_smoke_context_desc_default();
    desc.nx = 48;
    desc.ny = 72;
    desc.nz = 48;
    desc.dt = 1.0f / 90.0f;
    desc.cell_size = 1.0f;
    desc.ambient_temperature = 0.0f;
    desc.density_buoyancy = 0.045f;
    desc.temperature_buoyancy = 0.12f;
    desc.vorticity_epsilon = 2.0f;
    desc.pressure_iterations = 80;
    desc.use_monotonic_cubic = 1u;

    VisualSimulationOfSmokeContext* context = visual_simulation_of_smoke_context_create(&desc);
    if (context == nullptr) {
        smoke_ok(VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME, "visual_simulation_of_smoke_context_create");
        return EXIT_FAILURE;
    }

    cudaStream_t stream = nullptr;
    float* density = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        visual_simulation_of_smoke_context_destroy(context);
        return EXIT_FAILURE;
    }

    const uint64_t scalar_bytes = visual_simulation_of_smoke_context_required_scalar_field_bytes(context);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&density), scalar_bytes), "cudaMalloc density")) {
        cudaStreamDestroy(stream);
        visual_simulation_of_smoke_context_destroy(context);
        return EXIT_FAILURE;
    }

    const ScalarField density_field{
        .grid =
            FieldGridDesc{
                .nx = desc.nx,
                .ny = desc.ny,
                .nz = desc.nz,
                .cell_size = desc.cell_size,
            },
        .values =
            FieldBufferView{
                .data = density,
                .size_bytes = scalar_bytes,
                .format = FIELD_FORMAT_F32,
                .memory_type = FIELD_MEMORY_TYPE_CUDA_DEVICE,
            },
    };

    if (!smoke_ok(visual_simulation_of_smoke_clear_async(context, stream), "visual_simulation_of_smoke_clear_async", context)) {
        cudaFree(density);
        cudaStreamDestroy(stream);
        visual_simulation_of_smoke_context_destroy(context);
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 16; ++frame) {
        nvtx3::scoped_range frame_range{"vsmoke.demo.frame"};
        VisualSimulationOfSmokeSourceDesc source{
            .center_x = static_cast<float>(desc.nx) * 0.5f,
            .center_y = static_cast<float>(desc.ny) * 0.18f,
            .center_z = static_cast<float>(desc.nz) * 0.5f,
            .radius = 4.5f,
            .density_amount = 0.85f,
            .temperature_amount = 1.35f,
            .velocity_x = 0.0f,
            .velocity_y = 1.2f,
            .velocity_z = 0.0f,
        };

        if (!smoke_ok(visual_simulation_of_smoke_add_source_async(context, &source, stream), "visual_simulation_of_smoke_add_source_async", context) ||
            !smoke_ok(visual_simulation_of_smoke_step_async(context, stream), "visual_simulation_of_smoke_step_async", context)) {
            cudaFree(density);
            cudaStreamDestroy(stream);
            visual_simulation_of_smoke_context_destroy(context);
            return EXIT_FAILURE;
        }
    }

    {
        nvtx3::scoped_range snapshot_range{"vsmoke.demo.snapshot"};
        if (!smoke_ok(visual_simulation_of_smoke_snapshot_density_async(context, &density_field, stream), "visual_simulation_of_smoke_snapshot_density_async", context) ||
            !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
            cudaFree(density);
            cudaStreamDestroy(stream);
            visual_simulation_of_smoke_context_destroy(context);
            return EXIT_FAILURE;
        }
    }

    {
        nvtx3::scoped_range copy_range{"vsmoke.demo.copy_to_host"};
        std::vector<float> host_density(static_cast<size_t>(scalar_bytes / sizeof(float)), 0.0f);
        if (!cuda_ok(cudaMemcpy(host_density.data(), density, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy density")) {
            cudaFree(density);
            cudaStreamDestroy(stream);
            visual_simulation_of_smoke_context_destroy(context);
            return EXIT_FAILURE;
        }

        const float total_density = std::accumulate(host_density.begin(), host_density.end(), 0.0f);
        const float peak_density = host_density.empty() ? 0.0f : *std::max_element(host_density.begin(), host_density.end());

        std::cout << "visual-simulation-of-smoke-app\n";
        std::cout << "grid: " << desc.nx << " x " << desc.ny << " x " << desc.nz << '\n';
        std::cout << "total density: " << total_density << '\n';
        std::cout << "peak density: " << peak_density << '\n';
    }

    cudaFree(density);
    cudaStreamDestroy(stream);
    visual_simulation_of_smoke_context_destroy(context);
    return EXIT_SUCCESS;
}
