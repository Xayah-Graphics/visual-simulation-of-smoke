module;

#include "visual-simulation-of-smoke-3d.h"
#include <cuda_runtime.h>

export module scene_plume;

import app;
import std;

export namespace scene_plume {

    class Scene {
    public:
        Scene();
        ~Scene();

        Scene(const Scene&)                = delete;
        Scene& operator=(const Scene&)     = delete;
        Scene(Scene&&) noexcept            = delete;
        Scene& operator=(Scene&&) noexcept = delete;

        [[nodiscard]] std::span<const app::FieldInfo> fields() const;
        [[nodiscard]] app::VisualizationSettings default_visualization() const;
        [[nodiscard]] app::SceneInfo info() const;
        [[nodiscard]] cudaStream_t stream() const;

        void rebuild();
        void step(int sim_steps);
        void export_field(uint32_t field_index, void* device_destination) const;
        void export_velocity(void* device_destination, float* host_destination) const;

    private:
        SmokeSimulationConfig config_{
            .nx                          = 96,
            .ny                          = 128,
            .nz                          = 96,
            .cell_size                   = 0.01f,
            .dt                          = 1.0f / 90.0f,
            .pressure_iterations         = 160,
            .ambient_temperature         = 0.0f,
            .buoyancy_density_factor     = 0.8f,
            .buoyancy_temperature_factor = 0.75f,
            .vorticity_confinement       = 0.20f,
            .scalar_advection_mode       = SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC,
            .flow_boundary =
                {
                    .x_minus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC},
                    .x_plus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC},
                    .y_minus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_NO_SLIP_WALL, .velocity_x = 0.0f, .velocity_y = 0.0f, .velocity_z = 0.0f, .pressure = 0.0f},
                    .y_plus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW, .velocity_x = 0.0f, .velocity_y = 0.0f, .velocity_z = 0.0f, .pressure = 0.0f},
                    .z_minus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC},
                    .z_plus = {.type = SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC},
                },
            .density_boundary =
                {
                    .x_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .x_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .y_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                    .y_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                    .z_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .z_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                },
            .temperature_boundary =
                {
                    .x_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .x_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .y_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                    .y_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE, .value = 0.0f},
                    .z_minus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                    .z_plus = {.type = SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC, .value = 0.0f},
                },
        };

        cudaStream_t stream_             = nullptr;
        SmokeSimulationContext context_  = nullptr;
        app::GridShape grid_{};

        float* density_source_device_     = nullptr;
        float* temperature_source_device_ = nullptr;
        float* force_x_device_            = nullptr;
        float* force_y_device_            = nullptr;
        float* force_z_device_            = nullptr;

        std::vector<float> density_source_host_{};
        std::vector<float> temperature_source_host_{};
        std::vector<float> emitter_weight_host_{};
        std::vector<float> force_x_host_{};
        std::vector<float> force_y_host_{};
        std::vector<float> force_z_host_{};
        std::mt19937 random_engine_{std::random_device{}()};
        app::SceneInfo info_{};
    };

} // namespace scene_plume
