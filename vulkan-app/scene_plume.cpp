module;

#include "visual-simulation-of-smoke-3d.h"
#include <cuda_runtime.h>

module scene_plume;

import app;
import std;

namespace scene_plume {

    namespace {

        struct PlumeFieldInfo {
            app::FieldInfo view{};
            uint32_t view_kind = SMOKE_SIMULATION_VIEW_DENSITY;
        };

        constexpr std::array field_catalog_storage{
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Density",
                        .preset =
                            {
                                .density_scale  = 1.35f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 2.8f,
                                .scalar_opacity = 5.4f,
                                .scalar_low_r   = 0.03f,
                                .scalar_low_g   = 0.04f,
                                .scalar_low_b   = 0.07f,
                                .scalar_high_r  = 0.95f,
                                .scalar_high_g  = 0.90f,
                                .scalar_high_b  = 0.82f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_DENSITY,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Temperature",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 14.0f,
                                .scalar_opacity = 2.6f,
                                .scalar_low_r   = 0.06f,
                                .scalar_low_g   = 0.06f,
                                .scalar_low_b   = 0.15f,
                                .scalar_high_r  = 0.98f,
                                .scalar_high_g  = 0.62f,
                                .scalar_high_b  = 0.14f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_TEMPERATURE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Velocity Magnitude",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 1.2f,
                                .scalar_opacity = 2.4f,
                                .scalar_low_r   = 0.05f,
                                .scalar_low_g   = 0.10f,
                                .scalar_low_b   = 0.22f,
                                .scalar_high_r  = 0.32f,
                                .scalar_high_g  = 0.90f,
                                .scalar_high_b  = 1.0f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_FLOW_VELOCITY_MAGNITUDE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Vorticity Magnitude",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 12.0f,
                                .scalar_opacity = 2.7f,
                                .scalar_low_r   = 0.08f,
                                .scalar_low_g   = 0.05f,
                                .scalar_low_b   = 0.16f,
                                .scalar_high_r  = 0.98f,
                                .scalar_high_g  = 0.42f,
                                .scalar_high_b  = 0.14f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_FLOW_VORTICITY_MAGNITUDE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Pressure",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -4.0f,
                                .scalar_max     = 4.0f,
                                .scalar_opacity = 2.2f,
                                .scalar_low_r   = 0.08f,
                                .scalar_low_g   = 0.20f,
                                .scalar_low_b   = 0.58f,
                                .scalar_high_r  = 0.96f,
                                .scalar_high_g  = 0.56f,
                                .scalar_high_b  = 0.16f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_FLOW_PRESSURE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Divergence",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = -2.0f,
                                .scalar_max     = 2.0f,
                                .scalar_opacity = 2.2f,
                                .scalar_low_r   = 0.05f,
                                .scalar_low_g   = 0.12f,
                                .scalar_low_b   = 0.48f,
                                .scalar_high_r  = 0.94f,
                                .scalar_high_g  = 0.28f,
                                .scalar_high_b  = 0.22f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_FLOW_DIVERGENCE,
            },
            PlumeFieldInfo{
                .view =
                    {
                        .label = "Occupancy",
                        .preset =
                            {
                                .density_scale  = 1.0f,
                                .scalar_min     = 0.0f,
                                .scalar_max     = 1.0f,
                                .scalar_opacity = 4.0f,
                                .scalar_low_r   = 0.0f,
                                .scalar_low_g   = 0.0f,
                                .scalar_low_b   = 0.0f,
                                .scalar_high_r  = 0.94f,
                                .scalar_high_g  = 0.18f,
                                .scalar_high_b  = 0.12f,
                            },
                    },
                .view_kind = SMOKE_SIMULATION_VIEW_OCCUPANCY,
            },
        };

        constexpr auto field_views = [] {
            std::array<app::FieldInfo, field_catalog_storage.size()> result{};
            for (size_t i = 0; i < result.size(); ++i) result[i] = field_catalog_storage[i].view;
            return result;
        }();

        void check_cuda(const cudaError_t status, const std::string_view what) {
            if (status == cudaSuccess) return;
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }

        void check_smoke(const SmokeSimulationResult code, const std::string_view what) {
            if (code == SMOKE_SIMULATION_RESULT_OK) return;
            throw std::runtime_error(std::string(what) + " failed (" + std::to_string(static_cast<int>(code)) + ")");
        }

    } // namespace

    Scene::Scene() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    Scene::~Scene() {
        if (context_ != nullptr) smoke_simulation_destroy_context_cuda(context_);
        if (density_source_device_ != nullptr) cudaFree(density_source_device_);
        if (temperature_source_device_ != nullptr) cudaFree(temperature_source_device_);
        if (force_x_device_ != nullptr) cudaFree(force_x_device_);
        if (force_y_device_ != nullptr) cudaFree(force_y_device_);
        if (force_z_device_ != nullptr) cudaFree(force_z_device_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    std::span<const app::FieldInfo> Scene::fields() const {
        return std::span<const app::FieldInfo>{field_views};
    }

    app::VisualizationSettings Scene::default_visualization() const {
        app::VisualizationSettings settings{
            .view_mode           = app::ViewMode::Volume,
            .plane_axis          = app::PlaneAxis::XY,
            .march_steps         = 128,
            .slice_position      = 0.5f,
            .show_velocity_plane = false,
            .background_bottom_r = 0.0f,
            .background_bottom_g = 0.0f,
            .background_bottom_b = 0.0f,
            .background_top_r    = 0.0f,
            .background_top_g    = 0.0f,
            .background_top_b    = 0.0f,
        };
        app::apply_field_preset(settings, field_catalog_storage[0].view.preset);
        return settings;
    }

    app::SceneInfo Scene::info() const {
        return info_;
    }

    cudaStream_t Scene::stream() const {
        return stream_;
    }

    void Scene::rebuild() {
        if (context_ != nullptr) check_smoke(smoke_simulation_destroy_context_cuda(context_), "smoke_simulation_destroy_context_cuda");
        if (density_source_device_ != nullptr) cudaFree(density_source_device_);
        if (temperature_source_device_ != nullptr) cudaFree(temperature_source_device_);
        if (force_x_device_ != nullptr) cudaFree(force_x_device_);
        if (force_y_device_ != nullptr) cudaFree(force_y_device_);
        if (force_z_device_ != nullptr) cudaFree(force_z_device_);

        context_                    = nullptr;
        density_source_device_      = nullptr;
        temperature_source_device_  = nullptr;
        force_x_device_             = nullptr;
        force_y_device_             = nullptr;
        force_z_device_             = nullptr;

        const SmokeSimulationContextCreateDesc create_desc{
            .config              = config_,
            .stream              = stream_,
            .initial_density     = 0.0f,
            .initial_temperature = config_.ambient_temperature,
        };
        check_smoke(smoke_simulation_create_context_cuda(&create_desc, &context_), "smoke_simulation_create_context_cuda");

        const auto nx           = config_.nx;
        const auto ny           = config_.ny;
        const auto nz           = config_.nz;
        const auto cell_count   = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
        const auto scalar_bytes = cell_count * sizeof(float);
        const float h           = config_.cell_size;
        const float extent_x    = static_cast<float>(nx) * h;
        const float extent_y    = static_cast<float>(ny) * h;
        const float extent_z    = static_cast<float>(nz) * h;
        grid_                   = {
            .nx        = static_cast<uint32_t>(nx),
            .ny        = static_cast<uint32_t>(ny),
            .nz        = static_cast<uint32_t>(nz),
            .cell_size = h,
        };

        density_source_host_.assign(cell_count, 0.0f);
        temperature_source_host_.assign(cell_count, 0.0f);
        emitter_weight_host_.assign(cell_count, 0.0f);
        force_x_host_.assign(cell_count, 0.0f);
        force_y_host_.assign(cell_count, 0.0f);
        force_z_host_.assign(cell_count, 0.0f);

        const float emitter_x = extent_x * 0.50f;
        const float emitter_y = extent_y * 0.10f;
        const float emitter_z = extent_z * 0.50f;
        const float emitter_rx = extent_x * 0.12f;
        const float emitter_ry = extent_y * 0.08f;
        const float emitter_rz = extent_z * 0.12f;

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const auto index = static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                    const float px   = (static_cast<float>(x) + 0.5f) * h;
                    const float py   = (static_cast<float>(y) + 0.5f) * h;
                    const float pz   = (static_cast<float>(z) + 0.5f) * h;
                    const float dx   = (px - emitter_x) / emitter_rx;
                    const float dy   = (py - emitter_y) / emitter_ry;
                    const float dz   = (pz - emitter_z) / emitter_rz;
                    const float r2   = dx * dx + dy * dy + dz * dz;
                    if (r2 > 1.0f) continue;
                    const float plume = std::exp(-2.2f * r2);
                    emitter_weight_host_[index]       = plume;
                    density_source_host_[index]     = 18.0f * plume;
                    temperature_source_host_[index] = 34.0f * plume;
                }
            }
        }

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&density_source_device_), scalar_bytes), "cudaMalloc density_source_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&temperature_source_device_), scalar_bytes), "cudaMalloc temperature_source_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_x_device_), scalar_bytes), "cudaMalloc force_x_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_y_device_), scalar_bytes), "cudaMalloc force_y_device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&force_z_device_), scalar_bytes), "cudaMalloc force_z_device");

        check_cuda(cudaMemcpyAsync(density_source_device_, density_source_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync density_source_device");
        check_cuda(cudaMemcpyAsync(temperature_source_device_, temperature_source_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync temperature_source_device");
        check_cuda(cudaMemcpyAsync(force_x_device_, force_x_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_x_device");
        check_cuda(cudaMemcpyAsync(force_y_device_, force_y_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_y_device");
        check_cuda(cudaMemcpyAsync(force_z_device_, force_z_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync force_z_device");
        check_smoke(smoke_simulation_update_density_source_cuda(context_, density_source_device_), "smoke_simulation_update_density_source_cuda");
        check_smoke(smoke_simulation_update_temperature_source_cuda(context_, temperature_source_device_), "smoke_simulation_update_temperature_source_cuda");
        check_smoke(smoke_simulation_update_force_cuda(context_, force_x_device_, force_y_device_, force_z_device_), "smoke_simulation_update_force_cuda");

        info_ = {
            .grid              = grid_,
            .dt                = config_.dt,
            .step_count        = 0,
            .last_step_call_ms = 0.0,
        };
    }

    void Scene::step(const int sim_steps) {
        if (sim_steps <= 0) return;

        constexpr float base_force               = 22.0f;
        constexpr float min_speed_scale          = 0.82f;
        constexpr float max_speed_scale          = 1.20f;
        constexpr float max_lateral_direction    = 0.28f;
        std::uniform_real_distribution<float> speed_distribution{min_speed_scale, max_speed_scale};
        std::uniform_real_distribution<float> direction_distribution{-max_lateral_direction, max_lateral_direction};

        for (int step_index = 0; step_index < sim_steps; ++step_index) {
            const auto begin = std::chrono::steady_clock::now();

            const float lateral_x = direction_distribution(random_engine_);
            const float lateral_z = direction_distribution(random_engine_);
            const float speed     = base_force * speed_distribution(random_engine_);
            const float direction_length = std::sqrt(lateral_x * lateral_x + 1.0f + lateral_z * lateral_z);
            const float direction_x      = lateral_x / direction_length;
            const float direction_y      = 1.0f / direction_length;
            const float direction_z      = lateral_z / direction_length;
            const float force_x          = speed * direction_x;
            const float force_y          = speed * direction_y;
            const float force_z          = speed * direction_z;

            for (size_t i = 0; i < emitter_weight_host_.size(); ++i) {
                const float emitter_weight = emitter_weight_host_[i];
                force_x_host_[i] = emitter_weight * force_x;
                force_y_host_[i] = emitter_weight * force_y;
                force_z_host_[i] = emitter_weight * force_z;
            }

            const auto scalar_bytes = emitter_weight_host_.size() * sizeof(float);
            check_cuda(cudaMemcpyAsync(force_x_device_, force_x_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync step force_x_device");
            check_cuda(cudaMemcpyAsync(force_y_device_, force_y_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync step force_y_device");
            check_cuda(cudaMemcpyAsync(force_z_device_, force_z_host_.data(), scalar_bytes, cudaMemcpyHostToDevice, stream_), "cudaMemcpyAsync step force_z_device");
            check_smoke(smoke_simulation_update_force_cuda(context_, force_x_device_, force_y_device_, force_z_device_), "smoke_simulation_update_force_cuda");
            check_smoke(smoke_simulation_step_cuda(context_), "smoke_simulation_step_cuda");
            info_.last_step_call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            ++info_.step_count;
        }
    }

    void Scene::export_field(const uint32_t field_index, void* const device_destination) const {
        const auto& field = field_catalog_storage[(std::min)(static_cast<size_t>(field_index), field_catalog_storage.size() - 1)];
        SmokeSimulationView view{};
        const SmokeSimulationViewRequest view_request{
            .kind            = field.view_kind,
            .consumer_stream = stream_,
        };
        check_smoke(smoke_simulation_get_view_cuda(context_, &view_request, &view), "smoke_simulation_get_view_cuda");
        check_cuda(cudaMemcpyAsync(device_destination, view.data0, static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz) * sizeof(float), cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync export field");
    }

    void Scene::export_velocity(void* const device_destination, float* const host_destination) const {
        SmokeSimulationView view{};
        const SmokeSimulationViewRequest view_request{
            .kind            = SMOKE_SIMULATION_VIEW_FLOW_VELOCITY,
            .consumer_stream = stream_,
        };
        check_smoke(smoke_simulation_get_view_cuda(context_, &view_request, &view), "smoke_simulation_get_view_cuda");
        const auto scalar_bytes = static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz) * sizeof(float);
        if (device_destination != nullptr) {
            auto* destination = static_cast<float*>(device_destination);
            check_cuda(cudaMemcpyAsync(destination, view.data0, scalar_bytes, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync velocity_x export");
            check_cuda(cudaMemcpyAsync(destination + static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz), view.data1, scalar_bytes, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync velocity_y export");
            check_cuda(cudaMemcpyAsync(destination + static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz) * 2u, view.data2, scalar_bytes, cudaMemcpyDeviceToDevice, stream_), "cudaMemcpyAsync velocity_z export");
        }
        if (host_destination == nullptr) return;
        check_cuda(cudaMemcpyAsync(host_destination, view.data0, scalar_bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync velocity_x host");
        check_cuda(cudaMemcpyAsync(host_destination + static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz), view.data1, scalar_bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync velocity_y host");
        check_cuda(cudaMemcpyAsync(host_destination + static_cast<size_t>(grid_.nx) * static_cast<size_t>(grid_.ny) * static_cast<size_t>(grid_.nz) * 2u, view.data2, scalar_bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync velocity_z host");
    }

} // namespace scene_plume
