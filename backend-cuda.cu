#include "visual-simulation-of-smoke-3d.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace smoke_simulation {

    enum SmokeScalarFieldKind : uint32_t {
        SMOKE_FIELD_DENSITY     = 0,
        SMOKE_FIELD_TEMPERATURE = 1,
    };

    enum SmokeVectorFieldKind : uint32_t {
        SMOKE_VECTOR_FORCE          = 0,
        SMOKE_VECTOR_SOLID_VELOCITY = 1,
    };

    struct ContextStorage {
        SmokeSimulationConfig config{};
        cudaStream_t stream = nullptr;
        dim3 block{};
        dim3 cells{};
        dim3 velocity_x_cells{};
        dim3 velocity_y_cells{};
        dim3 velocity_z_cells{};
        std::uint64_t cell_count       = 0;
        std::uint64_t velocity_x_count = 0;
        std::uint64_t velocity_y_count = 0;
        std::uint64_t velocity_z_count = 0;
        std::size_t cell_bytes         = 0;
        std::size_t velocity_x_bytes   = 0;
        std::size_t velocity_y_bytes   = 0;
        std::size_t velocity_z_bytes   = 0;
        bool owns_stream               = false;

        struct StepRuntimeStorage {
            int pressure_anchor                = 0;
            bool occupancy_dirty              = false;
            std::vector<uint8_t> occupancy_host{};
        } step_runtime{};

        struct DeviceBuffers {
            struct Flow {
                float* velocity_x          = nullptr;
                float* velocity_y          = nullptr;
                float* velocity_z          = nullptr;
                float* temp_velocity_x     = nullptr;
                float* temp_velocity_y     = nullptr;
                float* temp_velocity_z     = nullptr;
                float* centered_velocity_x = nullptr;
                float* centered_velocity_y = nullptr;
                float* centered_velocity_z = nullptr;
                float* velocity_magnitude  = nullptr;
                float* pressure            = nullptr;
                float* pressure_rhs        = nullptr;
                float* divergence          = nullptr;
                float* vorticity_x         = nullptr;
                float* vorticity_y         = nullptr;
                float* vorticity_z         = nullptr;
                float* vorticity_magnitude = nullptr;
                float* force_x             = nullptr;
                float* force_y             = nullptr;
                float* force_z             = nullptr;
            } flow{};

            struct ScalarField {
                SmokeScalarFieldKind kind = SMOKE_FIELD_DENSITY;
                float* data               = nullptr;
                float* temp               = nullptr;
                float* source             = nullptr;
            };

            struct VectorField {
                SmokeVectorFieldKind kind = SMOKE_VECTOR_FORCE;
                float* data_x             = nullptr;
                float* data_y             = nullptr;
                float* data_z             = nullptr;
            };

            std::vector<ScalarField> scalar_fields{};
            std::vector<VectorField> vector_fields{};
            float* solid_temperature = nullptr;
            float* occupancy_float   = nullptr;
            uint8_t* occupancy       = nullptr;
        } device{};
    };

    void check_cuda(cudaError_t status, const char* what);

    __host__ __device__ std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
    }

    __host__ __device__ std::uint64_t index_velocity_x(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx + 1) + static_cast<std::uint64_t>(i);
    }

    __host__ __device__ std::uint64_t index_velocity_y(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
    }

    __host__ __device__ std::uint64_t index_velocity_z(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
    }

    dim3 grid_for(const int sx, const int sy, const int sz, const dim3 block) {
        return dim3(
            static_cast<unsigned>((sx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((sy + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((sz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    __host__ __device__ int wrap_index(int value, const int size) {
        if (size <= 0) return 0;
        value %= size;
        if (value < 0) value += size;
        return value;
    }

    __host__ __device__ bool cell_in_bounds(const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __host__ __device__ bool resolve_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) x = wrap_index(x, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) y = wrap_index(y, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ bool load_occupancy(const uint8_t* occupancy, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (occupancy == nullptr) return false;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return true;
        return occupancy[index_3d(x, y, z, nx, ny)] != 0;
    }

    __device__ float load_scalar(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_velocity_x(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i > nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0.0f;
        return field[index_velocity_x(i, j, k, nx, ny)];
    }

    __device__ float load_velocity_y(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i >= nx || j < 0 || j > ny || k < 0 || k >= nz) return 0.0f;
        return field[index_velocity_y(i, j, k, nx, ny)];
    }

    __device__ float load_velocity_z(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k > nz) return 0.0f;
        return field[index_velocity_z(i, j, k, nx, ny)];
    }

    __device__ float monotonic_cubic_1d(const float p0, const float p1, const float p2, const float p3, const float t) {
        const float delta = p2 - p1;
        float m1          = 0.5f * (p2 - p0);
        float m2          = 0.5f * (p3 - p1);
        if (fabsf(delta) < 1.0e-6f) {
            m1 = 0.0f;
            m2 = 0.0f;
        } else {
            if (m1 * delta <= 0.0f) m1 = 0.0f;
            if (m2 * delta <= 0.0f) m2 = 0.0f;
        }
        const float t2 = t * t;
        const float t3 = t2 * t;
        return (2.0f * t3 - 3.0f * t2 + 1.0f) * p1 + (t3 - 2.0f * t2 + t) * m1 + (-2.0f * t3 + 3.0f * t2) * p2 + (t3 - t2) * m2;
    }

    __device__ float sample_scalar_linear(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        const int x0   = static_cast<int>(floorf(gx));
        const int y0   = static_cast<int>(floorf(gy));
        const int z0   = static_cast<int>(floorf(gz));
        const int x1   = x0 + 1;
        const int y1   = y0 + 1;
        const int z1   = z0 + 1;
        const float tx = gx - static_cast<float>(x0);
        const float ty = gy - static_cast<float>(y0);
        const float tz = gz - static_cast<float>(z0);

        const float c000 = load_scalar(field, x0, y0, z0, nx, ny, nz, boundary);
        const float c100 = load_scalar(field, x1, y0, z0, nx, ny, nz, boundary);
        const float c010 = load_scalar(field, x0, y1, z0, nx, ny, nz, boundary);
        const float c110 = load_scalar(field, x1, y1, z0, nx, ny, nz, boundary);
        const float c001 = load_scalar(field, x0, y0, z1, nx, ny, nz, boundary);
        const float c101 = load_scalar(field, x1, y0, z1, nx, ny, nz, boundary);
        const float c011 = load_scalar(field, x0, y1, z1, nx, ny, nz, boundary);
        const float c111 = load_scalar(field, x1, y1, z1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_scalar_cubic(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        const int x1   = static_cast<int>(floorf(gx));
        const int y1   = static_cast<int>(floorf(gy));
        const int z1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(x1);
        const float ty = gy - static_cast<float>(y1);
        const float tz = gz - static_cast<float>(z1);

        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int yy = y1 + dy - 1;
                const int zz = z1 + dz - 1;
                const float p0 = load_scalar(field, x1 - 1, yy, zz, nx, ny, nz, boundary);
                const float p1 = load_scalar(field, x1, yy, zz, nx, ny, nz, boundary);
                const float p2 = load_scalar(field, x1 + 1, yy, zz, nx, ny, nz, boundary);
                const float p3 = load_scalar(field, x1 + 2, yy, zz, nx, ny, nz, boundary);
                y_samples[dy] = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_velocity_x(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

        const float gx = x / h;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        const int i0   = static_cast<int>(floorf(gx));
        const int j0   = static_cast<int>(floorf(gy));
        const int k0   = static_cast<int>(floorf(gz));
        const int i1   = i0 + 1;
        const int j1   = j0 + 1;
        const int k1   = k0 + 1;
        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);
        const float tz = gz - static_cast<float>(k0);

        const float c000 = load_velocity_x(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_velocity_x(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_velocity_x(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_velocity_x(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_velocity_x(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_velocity_x(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_velocity_x(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_velocity_x(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_velocity_y(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

        const float gx = x / h - 0.5f;
        const float gy = y / h;
        const float gz = z / h - 0.5f;
        const int i0   = static_cast<int>(floorf(gx));
        const int j0   = static_cast<int>(floorf(gy));
        const int k0   = static_cast<int>(floorf(gz));
        const int i1   = i0 + 1;
        const int j1   = j0 + 1;
        const int k1   = k0 + 1;
        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);
        const float tz = gz - static_cast<float>(k0);

        const float c000 = load_velocity_y(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_velocity_y(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_velocity_y(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_velocity_y(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_velocity_y(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_velocity_y(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_velocity_y(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_velocity_y(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_velocity_z(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }

        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h;
        const int i0   = static_cast<int>(floorf(gx));
        const int j0   = static_cast<int>(floorf(gy));
        const int k0   = static_cast<int>(floorf(gz));
        const int i1   = i0 + 1;
        const int j1   = j0 + 1;
        const int k1   = k0 + 1;
        const float tx = gx - static_cast<float>(i0);
        const float ty = gy - static_cast<float>(j0);
        const float tz = gz - static_cast<float>(k0);

        const float c000 = load_velocity_z(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_velocity_z(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_velocity_z(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_velocity_z(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_velocity_z(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_velocity_z(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_velocity_z(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_velocity_z(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h,
        const SmokeSimulationBoundaryConfig boundary) {
        return make_float3(
            sample_velocity_x(velocity_x, x, y, z, nx, ny, nz, h, boundary),
            sample_velocity_y(velocity_y, x, y, z, nx, ny, nz, h, boundary),
            sample_velocity_z(velocity_z, x, y, z, nx, ny, nz, h, boundary));
    }

    __device__ float3 trace_particle_rk2(const float3 start, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const float dt, const int nx, const int ny, const int nz, const float h,
        const SmokeSimulationBoundaryConfig boundary) {
        const float3 velocity0 = sample_velocity(velocity_x, velocity_y, velocity_z, start.x, start.y, start.z, nx, ny, nz, h, boundary);
        const float3 mid       = make_float3(start.x - 0.5f * dt * velocity0.x, start.y - 0.5f * dt * velocity0.y, start.z - 0.5f * dt * velocity0.z);
        const float3 velocity1 = sample_velocity(velocity_x, velocity_y, velocity_z, mid.x, mid.y, mid.z, nx, ny, nz, h, boundary);
        float3 traced          = make_float3(start.x - dt * velocity1.x, start.y - dt * velocity1.y, start.z - dt * velocity1.z);
        float end_x            = traced.x;
        float end_y            = traced.y;
        float end_z            = traced.z;
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            end_x                = fmodf(end_x, extent_x);
            if (end_x < 0.0f) end_x += extent_x;
        }
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            end_y                = fmodf(end_y, extent_y);
            if (end_y < 0.0f) end_y += extent_y;
        }
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            end_z                = fmodf(end_z, extent_z);
            if (end_z < 0.0f) end_z += extent_z;
        }

        bool traced_hits_solid = end_x < 0.0f || end_x > static_cast<float>(nx) * h || end_y < 0.0f || end_y > static_cast<float>(ny) * h || end_z < 0.0f || end_z > static_cast<float>(nz) * h;
        if (!traced_hits_solid && occupancy != nullptr) {
            int end_cell_x = static_cast<int>(floorf(end_x / h));
            int end_cell_y = static_cast<int>(floorf(end_y / h));
            int end_cell_z = static_cast<int>(floorf(end_z / h));
            if (end_cell_x == nx) end_cell_x = nx - 1;
            if (end_cell_y == ny) end_cell_y = ny - 1;
            if (end_cell_z == nz) end_cell_z = nz - 1;
            traced_hits_solid = !cell_in_bounds(end_cell_x, end_cell_y, end_cell_z, nx, ny, nz) || occupancy[index_3d(end_cell_x, end_cell_y, end_cell_z, nx, ny)] != 0;
        }
        if (!traced_hits_solid) return traced;

        float lo = 0.0f;
        float hi = 1.0f;
        for (int iteration = 0; iteration < 10; ++iteration) {
            const float mid_t = 0.5f * (lo + hi);
            float test_x      = start.x + (traced.x - start.x) * mid_t;
            float test_y      = start.y + (traced.y - start.y) * mid_t;
            float test_z      = start.z + (traced.z - start.z) * mid_t;
            if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) {
                const float extent_x = static_cast<float>(nx) * h;
                test_x               = fmodf(test_x, extent_x);
                if (test_x < 0.0f) test_x += extent_x;
            }
            if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) {
                const float extent_y = static_cast<float>(ny) * h;
                test_y               = fmodf(test_y, extent_y);
                if (test_y < 0.0f) test_y += extent_y;
            }
            if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) {
                const float extent_z = static_cast<float>(nz) * h;
                test_z               = fmodf(test_z, extent_z);
                if (test_z < 0.0f) test_z += extent_z;
            }

            bool test_hits_solid = test_x < 0.0f || test_x > static_cast<float>(nx) * h || test_y < 0.0f || test_y > static_cast<float>(ny) * h || test_z < 0.0f || test_z > static_cast<float>(nz) * h;
            if (!test_hits_solid && occupancy != nullptr) {
                int test_cell_x = static_cast<int>(floorf(test_x / h));
                int test_cell_y = static_cast<int>(floorf(test_y / h));
                int test_cell_z = static_cast<int>(floorf(test_z / h));
                if (test_cell_x == nx) test_cell_x = nx - 1;
                if (test_cell_y == ny) test_cell_y = ny - 1;
                if (test_cell_z == nz) test_cell_z = nz - 1;
                test_hits_solid = !cell_in_bounds(test_cell_x, test_cell_y, test_cell_z, nx, ny, nz) || occupancy[index_3d(test_cell_x, test_cell_y, test_cell_z, nx, ny)] != 0;
            }
            if (test_hits_solid) hi = mid_t;
            else lo = mid_t;
        }
        traced.x = start.x + (traced.x - start.x) * lo;
        traced.y = start.y + (traced.y - start.y) * lo;
        traced.z = start.z + (traced.z - start.z) * lo;
        return traced;
    }

    __global__ void fill_float_kernel(float* field, const float value, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        field[index] = value;
    }

    __global__ void copy_u8_to_float_kernel(float* destination, const uint8_t* source, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        destination[index] = source != nullptr ? static_cast<float>(source[index]) : 0.0f;
    }

    __global__ void add_source_kernel(float* destination, const float* current, const float* source, const float dt, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        const float source_value = source != nullptr ? source[index] : 0.0f;
        destination[index]       = current[index] + dt * source_value;
    }

    __global__ void compute_center_velocity_kernel(float* cell_x, float* cell_y, float* cell_z, const float* velocity_x, const float* velocity_y, const float* velocity_z, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        cell_x[index]    = 0.5f * (velocity_x[index_velocity_x(x, y, z, nx, ny)] + velocity_x[index_velocity_x(x + 1, y, z, nx, ny)]);
        cell_y[index]    = 0.5f * (velocity_y[index_velocity_y(x, y, z, nx, ny)] + velocity_y[index_velocity_y(x, y + 1, z, nx, ny)]);
        cell_z[index]    = 0.5f * (velocity_z[index_velocity_z(x, y, z, nx, ny)] + velocity_z[index_velocity_z(x, y, z + 1, nx, ny)]);
    }

    __global__ void compute_vorticity_kernel(float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* cell_x, const float* cell_y, const float* cell_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h,
        const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = index_3d(x, y, z, nx, ny);
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) {
            omega_x[index]         = 0.0f;
            omega_y[index]         = 0.0f;
            omega_z[index]         = 0.0f;
            omega_magnitude[index] = 0.0f;
            return;
        }

        const float dvz_dy = 0.5f * (load_scalar(cell_z, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(cell_z, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float dvy_dz = 0.5f * (load_scalar(cell_y, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(cell_y, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float dvx_dz = 0.5f * (load_scalar(cell_x, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(cell_x, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float dvz_dx = 0.5f * (load_scalar(cell_z, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(cell_z, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float dvy_dx = 0.5f * (load_scalar(cell_y, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(cell_y, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float dvx_dy = 0.5f * (load_scalar(cell_x, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(cell_x, x, y - 1, z, nx, ny, nz, boundary)) / h;

        const float wx = dvz_dy - dvy_dz;
        const float wy = dvx_dz - dvz_dx;
        const float wz = dvy_dx - dvx_dy;

        omega_x[index]         = wx;
        omega_y[index]         = wy;
        omega_z[index]         = wz;
        omega_magnitude[index] = sqrtf(wx * wx + wy * wy + wz * wz);
    }

    __global__ void seed_force_kernel(float* force_x, float* force_y, float* force_z, const float* source_x, const float* source_y, const float* source_z, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        force_x[index] = source_x != nullptr ? source_x[index] : 0.0f;
        force_y[index] = source_y != nullptr ? source_y[index] : 0.0f;
        force_z[index] = source_z != nullptr ? source_z[index] : 0.0f;
    }

    __global__ void add_buoyancy_kernel(float* force_y, const float* density, const float* temperature, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor,
        const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;
        const auto index = index_3d(x, y, z, nx, ny);
        force_y[index] += -density_factor * density[index] + temperature_factor * (temperature[index] - ambient_temperature);
    }

    __global__ void add_confinement_kernel(float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h,
        const float epsilon, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;

        const float grad_x   = 0.5f * (load_scalar(omega_magnitude, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(omega_magnitude, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float grad_y   = 0.5f * (load_scalar(omega_magnitude, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(omega_magnitude, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float grad_z   = 0.5f * (load_scalar(omega_magnitude, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(omega_magnitude, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
        if (grad_mag < 1.0e-6f) return;

        const float inv_grad      = 1.0f / grad_mag;
        const float normal_x      = grad_x * inv_grad;
        const float normal_y      = grad_y * inv_grad;
        const float normal_z      = grad_z * inv_grad;
        const auto index          = index_3d(x, y, z, nx, ny);
        const float wx            = omega_x[index];
        const float wy            = omega_y[index];
        const float wz            = omega_z[index];
        const float confinement_x = epsilon * h * (normal_y * wz - normal_z * wy);
        const float confinement_y = epsilon * h * (normal_z * wx - normal_x * wz);
        const float confinement_z = epsilon * h * (normal_x * wy - normal_y * wx);

        force_x[index] += confinement_x;
        force_y[index] += confinement_y;
        force_z[index] += confinement_z;
    }

    __global__ void add_center_forces_to_velocity_x_kernel(float* velocity_x, const float* force_x, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        float sum    = 0.0f;
        float weight = 0.0f;
        if (i > 0) {
            sum += force_x[index_3d(i - 1, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (i < nx) {
            sum += force_x[index_3d(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) velocity_x[index_velocity_x(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __global__ void add_center_forces_to_velocity_y_kernel(float* velocity_y, const float* force_y, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        float sum    = 0.0f;
        float weight = 0.0f;
        if (j > 0) {
            sum += force_y[index_3d(i, j - 1, k, nx, ny)];
            weight += 1.0f;
        }
        if (j < ny) {
            sum += force_y[index_3d(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) velocity_y[index_velocity_y(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __global__ void add_center_forces_to_velocity_z_kernel(float* velocity_z, const float* force_z, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        float sum    = 0.0f;
        float weight = 0.0f;
        if (k > 0) {
            sum += force_z[index_3d(i, j, k - 1, nx, ny)];
            weight += 1.0f;
        }
        if (k < nz) {
            sum += force_z[index_3d(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) velocity_z[index_velocity_z(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __device__ float solid_velocity_value(const float* solid_velocity, const uint8_t* occupancy, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (solid_velocity == nullptr || occupancy == nullptr) return 0.0f;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        if (occupancy[index_3d(x, y, z, nx, ny)] == 0) return 0.0f;
        return solid_velocity[index_3d(x, y, z, nx, ny)];
    }

    __global__ void enforce_velocity_x_boundaries_kernel(float* velocity_x, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[index_velocity_x(i, j, k, nx, ny)];
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_FIXED && (i == 0 || i == nx)) {
            face = 0.0f;
            return;
        }
        if (occupancy == nullptr) return;

        int left_x  = i - 1;
        int left_y  = j;
        int left_z  = k;
        int right_x = i;
        int right_y = j;
        int right_z = k;
        const bool has_left       = resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary);
        const bool has_right      = resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary);
        const bool left_occupied  = has_left && occupancy[index_3d(left_x, left_y, left_z, nx, ny)] != 0;
        const bool right_occupied = has_right && occupancy[index_3d(right_x, right_y, right_z, nx, ny)] != 0;
        if (!left_occupied && !right_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (left_occupied) {
            value += solid_velocity_value(solid_velocity_x, occupancy, left_x, left_y, left_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (right_occupied) {
            value += solid_velocity_value(solid_velocity_x, occupancy, right_x, right_y, right_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_velocity_y_boundaries_kernel(float* velocity_y, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[index_velocity_y(i, j, k, nx, ny)];
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_FIXED && (j == 0 || j == ny)) {
            face = 0.0f;
            return;
        }
        if (occupancy == nullptr) return;

        int down_x  = i;
        int down_y  = j - 1;
        int down_z  = k;
        int up_x    = i;
        int up_y    = j;
        int up_z    = k;
        const bool has_down       = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up         = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied  = has_down && occupancy[index_3d(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied    = has_up && occupancy[index_3d(up_x, up_y, up_z, nx, ny)] != 0;
        if (!down_occupied && !up_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (down_occupied) {
            value += solid_velocity_value(solid_velocity_y, occupancy, down_x, down_y, down_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (up_occupied) {
            value += solid_velocity_value(solid_velocity_y, occupancy, up_x, up_y, up_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void enforce_velocity_z_boundaries_kernel(float* velocity_z, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[index_velocity_z(i, j, k, nx, ny)];
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_FIXED && (k == 0 || k == nz)) {
            face = 0.0f;
            return;
        }
        if (occupancy == nullptr) return;

        int back_x   = i;
        int back_y   = j;
        int back_z   = k - 1;
        int front_x  = i;
        int front_y  = j;
        int front_z  = k;
        const bool has_back       = resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary);
        const bool has_front      = resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary);
        const bool back_occupied  = has_back && occupancy[index_3d(back_x, back_y, back_z, nx, ny)] != 0;
        const bool front_occupied = has_front && occupancy[index_3d(front_x, front_y, front_z, nx, ny)] != 0;
        if (!back_occupied && !front_occupied) return;

        float value  = 0.0f;
        float weight = 0.0f;
        if (back_occupied) {
            value += solid_velocity_value(solid_velocity_z, occupancy, back_x, back_y, back_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        if (front_occupied) {
            value += solid_velocity_value(solid_velocity_z, occupancy, front_x, front_y, front_z, nx, ny, nz, boundary);
            weight += 1.0f;
        }
        face = weight > 0.0f ? value / weight : 0.0f;
    }

    __global__ void sync_periodic_velocity_x_kernel(float* velocity_x, const int nx, const int ny, const int nz) {
        const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (j >= ny || k >= nz) return;
        velocity_x[index_velocity_x(nx, j, k, nx, ny)] = velocity_x[index_velocity_x(0, j, k, nx, ny)];
    }

    __global__ void sync_periodic_velocity_y_kernel(float* velocity_y, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || k >= nz) return;
        velocity_y[index_velocity_y(i, ny, k, nx, ny)] = velocity_y[index_velocity_y(i, 0, k, nx, ny)];
    }

    __global__ void sync_periodic_velocity_z_kernel(float* velocity_z, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || j >= ny) return;
        velocity_z[index_velocity_z(i, j, nz, nx, ny)] = velocity_z[index_velocity_z(i, j, 0, nx, ny)];
    }

    __global__ void advect_velocity_x_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt,
        const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;
        const float3 start  = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_x(i, j, k, nx, ny)] = sample_velocity_x(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_velocity_y_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt,
        const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;
        const float3 start  = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_y(i, j, k, nx, ny)] = sample_velocity_y(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_velocity_z_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt,
        const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;
        const float3 start  = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
        const float3 traced = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_z(i, j, k, nx, ny)] = sample_velocity_z(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_scalar_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt,
        const uint32_t advection_mode, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) {
            destination[index_3d(x, y, z, nx, ny)] = 0.0f;
            return;
        }
        const float3 start  = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_3d(x, y, z, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_scalar_cubic(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary)
                                                                                                                           : sample_scalar_linear(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void apply_solid_temperature_kernel(float* temperature, const uint8_t* occupancy, const float* solid_temperature, const int nx, const int ny, const int nz, const float ambient_temperature) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (index >= count) return;
        if (occupancy == nullptr || occupancy[index] == 0) return;
        temperature[index] = solid_temperature != nullptr ? solid_temperature[index] : ambient_temperature;
    }

    __global__ void boundary_fill_density_kernel(float* destination, const float* source, const uint8_t* occupancy, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = index_3d(x, y, z, nx, ny);
        if (occupancy == nullptr || occupancy[index] == 0) {
            destination[index] = source[index];
            return;
        }

        float sum    = 0.0f;
        float weight = 0.0f;
        const int offsets[6][3] = {
            {-1, 0, 0},
            {1, 0, 0},
            {0, -1, 0},
            {0, 1, 0},
            {0, 0, -1},
            {0, 0, 1},
        };
        for (const auto& offset : offsets) {
            int next_x = x + offset[0];
            int next_y = y + offset[1];
            int next_z = z + offset[2];
            if (!resolve_cell_coordinates(next_x, next_y, next_z, nx, ny, nz, boundary)) continue;
            const auto neighbor_index = index_3d(next_x, next_y, next_z, nx, ny);
            if (occupancy[neighbor_index] != 0) continue;
            sum += source[neighbor_index];
            weight += 1.0f;
        }
        destination[index] = weight > 0.0f ? sum / weight : 0.0f;
    }

    __global__ void compute_pressure_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (static_cast<int>(index) == pressure_anchor) {
            rhs[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            rhs[index] = 0.0f;
            return;
        }
        const float divergence = (velocity_x[index_velocity_x(x + 1, y, z, nx, ny)] - velocity_x[index_velocity_x(x, y, z, nx, ny)] + velocity_y[index_velocity_y(x, y + 1, z, nx, ny)] - velocity_y[index_velocity_y(x, y, z, nx, ny)] + velocity_z[index_velocity_z(x, y, z + 1, nx, ny)] -
                                     velocity_z[index_velocity_z(x, y, z, nx, ny)]) /
            h;
        rhs[index] = -divergence / dt;
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        if (occupancy != nullptr && occupancy[index] != 0) {
            divergence[index] = 0.0f;
            return;
        }
        divergence[index] = (velocity_x[index_velocity_x(x + 1, y, z, nx, ny)] - velocity_x[index_velocity_x(x, y, z, nx, ny)] + velocity_y[index_velocity_y(x, y + 1, z, nx, ny)] - velocity_y[index_velocity_y(x, y, z, nx, ny)] + velocity_z[index_velocity_z(x, y, z + 1, nx, ny)] -
                                 velocity_z[index_velocity_z(x, y, z, nx, ny)]) /
            h;
    }

    __global__ void rbgs_pressure_kernel(float* pressure, const float* rhs, const uint8_t* occupancy, const int pressure_anchor, const int parity, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;

        const auto index = index_3d(x, y, z, nx, ny);
        if (static_cast<int>(index) == pressure_anchor) {
            pressure[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            pressure[index] = 0.0f;
            return;
        }

        float diagonal = 0.0f;
        float sum      = 0.0f;
        const int offsets[6][3] = {
            {-1, 0, 0},
            {1, 0, 0},
            {0, -1, 0},
            {0, 1, 0},
            {0, 0, -1},
            {0, 0, 1},
        };
        for (const auto& offset : offsets) {
            int next_x = x + offset[0];
            int next_y = y + offset[1];
            int next_z = z + offset[2];
            if (!resolve_cell_coordinates(next_x, next_y, next_z, nx, ny, nz, boundary)) continue;
            const auto neighbor_index = index_3d(next_x, next_y, next_z, nx, ny);
            if (occupancy != nullptr && occupancy[neighbor_index] != 0) continue;
            diagonal += 1.0f;
            if (static_cast<int>(neighbor_index) == pressure_anchor) continue;
            sum += pressure[neighbor_index];
        }
        pressure[index] = diagonal > 0.0f ? (sum + rhs[index] * h * h) / diagonal : 0.0f;
    }

    __global__ void project_velocity_x_kernel(float* velocity_x, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[index_velocity_x(i, j, k, nx, ny)];
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_FIXED && (i == 0 || i == nx)) {
            face = 0.0f;
            return;
        }

        int left_x   = i - 1;
        int left_y   = j;
        int left_z   = k;
        int right_x  = i;
        int right_y  = j;
        int right_z  = k;
        const bool has_left       = resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary);
        const bool has_right      = resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary);
        const bool left_occupied  = has_left && occupancy != nullptr && occupancy[index_3d(left_x, left_y, left_z, nx, ny)] != 0;
        const bool right_occupied = has_right && occupancy != nullptr && occupancy[index_3d(right_x, right_y, right_z, nx, ny)] != 0;
        if (left_occupied || right_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (left_occupied) {
                value += solid_velocity_value(solid_velocity_x, occupancy, left_x, left_y, left_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            if (right_occupied) {
                value += solid_velocity_value(solid_velocity_x, occupancy, right_x, right_y, right_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_left && has_right) {
            const float pressure_right = pressure[index_3d(right_x, right_y, right_z, nx, ny)];
            const float pressure_left  = pressure[index_3d(left_x, left_y, left_z, nx, ny)];
            face -= dt * (pressure_right - pressure_left) / h;
        }
    }

    __global__ void project_velocity_y_kernel(float* velocity_y, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[index_velocity_y(i, j, k, nx, ny)];
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_FIXED && (j == 0 || j == ny)) {
            face = 0.0f;
            return;
        }

        int down_x   = i;
        int down_y   = j - 1;
        int down_z   = k;
        int up_x     = i;
        int up_y     = j;
        int up_z     = k;
        const bool has_down      = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up        = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied = has_down && occupancy != nullptr && occupancy[index_3d(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied   = has_up && occupancy != nullptr && occupancy[index_3d(up_x, up_y, up_z, nx, ny)] != 0;
        if (down_occupied || up_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (down_occupied) {
                value += solid_velocity_value(solid_velocity_y, occupancy, down_x, down_y, down_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            if (up_occupied) {
                value += solid_velocity_value(solid_velocity_y, occupancy, up_x, up_y, up_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_down && has_up) {
            const float pressure_up   = pressure[index_3d(up_x, up_y, up_z, nx, ny)];
            const float pressure_down = pressure[index_3d(down_x, down_y, down_z, nx, ny)];
            face -= dt * (pressure_up - pressure_down) / h;
        }
    }

    __global__ void project_velocity_z_kernel(float* velocity_z, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[index_velocity_z(i, j, k, nx, ny)];
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_FIXED && (k == 0 || k == nz)) {
            face = 0.0f;
            return;
        }

        int back_x   = i;
        int back_y   = j;
        int back_z   = k - 1;
        int front_x  = i;
        int front_y  = j;
        int front_z  = k;
        const bool has_back       = resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary);
        const bool has_front      = resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary);
        const bool back_occupied  = has_back && occupancy != nullptr && occupancy[index_3d(back_x, back_y, back_z, nx, ny)] != 0;
        const bool front_occupied = has_front && occupancy != nullptr && occupancy[index_3d(front_x, front_y, front_z, nx, ny)] != 0;
        if (back_occupied || front_occupied) {
            float value  = 0.0f;
            float weight = 0.0f;
            if (back_occupied) {
                value += solid_velocity_value(solid_velocity_z, occupancy, back_x, back_y, back_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            if (front_occupied) {
                value += solid_velocity_value(solid_velocity_z, occupancy, front_x, front_y, front_z, nx, ny, nz, boundary);
                weight += 1.0f;
            }
            face = weight > 0.0f ? value / weight : 0.0f;
            return;
        }
        if (has_back && has_front) {
            const float pressure_front = pressure[index_3d(front_x, front_y, front_z, nx, ny)];
            const float pressure_back  = pressure[index_3d(back_x, back_y, back_z, nx, ny)];
            face -= dt * (pressure_front - pressure_back) / h;
        }
    }

    __global__ void velocity_magnitude_kernel(float* destination, const float* velocity_x, const float* velocity_y, const float* velocity_z, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        const float vx     = velocity_x[index];
        const float vy     = velocity_y[index];
        const float vz     = velocity_z[index];
        destination[index] = sqrtf(vx * vx + vy * vy + vz * vz);
    }

    void destroy_context_resources(ContextStorage& context) {
        if (context.device.flow.velocity_x != nullptr) cudaFree(context.device.flow.velocity_x);
        if (context.device.flow.velocity_y != nullptr) cudaFree(context.device.flow.velocity_y);
        if (context.device.flow.velocity_z != nullptr) cudaFree(context.device.flow.velocity_z);
        if (context.device.flow.temp_velocity_x != nullptr) cudaFree(context.device.flow.temp_velocity_x);
        if (context.device.flow.temp_velocity_y != nullptr) cudaFree(context.device.flow.temp_velocity_y);
        if (context.device.flow.temp_velocity_z != nullptr) cudaFree(context.device.flow.temp_velocity_z);
        if (context.device.flow.centered_velocity_x != nullptr) cudaFree(context.device.flow.centered_velocity_x);
        if (context.device.flow.centered_velocity_y != nullptr) cudaFree(context.device.flow.centered_velocity_y);
        if (context.device.flow.centered_velocity_z != nullptr) cudaFree(context.device.flow.centered_velocity_z);
        if (context.device.flow.velocity_magnitude != nullptr) cudaFree(context.device.flow.velocity_magnitude);
        if (context.device.flow.pressure != nullptr) cudaFree(context.device.flow.pressure);
        if (context.device.flow.pressure_rhs != nullptr) cudaFree(context.device.flow.pressure_rhs);
        if (context.device.flow.divergence != nullptr) cudaFree(context.device.flow.divergence);
        if (context.device.flow.vorticity_x != nullptr) cudaFree(context.device.flow.vorticity_x);
        if (context.device.flow.vorticity_y != nullptr) cudaFree(context.device.flow.vorticity_y);
        if (context.device.flow.vorticity_z != nullptr) cudaFree(context.device.flow.vorticity_z);
        if (context.device.flow.vorticity_magnitude != nullptr) cudaFree(context.device.flow.vorticity_magnitude);
        if (context.device.flow.force_x != nullptr) cudaFree(context.device.flow.force_x);
        if (context.device.flow.force_y != nullptr) cudaFree(context.device.flow.force_y);
        if (context.device.flow.force_z != nullptr) cudaFree(context.device.flow.force_z);
        if (context.device.occupancy_float != nullptr) cudaFree(context.device.occupancy_float);
        if (context.device.occupancy != nullptr) cudaFree(context.device.occupancy);
        if (context.device.solid_temperature != nullptr) cudaFree(context.device.solid_temperature);
        context.device.flow = ContextStorage::DeviceBuffers::Flow{};
        for (auto& field : context.device.scalar_fields) {
            if (field.data != nullptr) cudaFree(field.data);
            if (field.temp != nullptr) cudaFree(field.temp);
            if (field.source != nullptr) cudaFree(field.source);
            field.data   = nullptr;
            field.temp   = nullptr;
            field.source = nullptr;
        }
        for (auto& field : context.device.vector_fields) {
            if (field.data_x != nullptr) cudaFree(field.data_x);
            if (field.data_y != nullptr) cudaFree(field.data_y);
            if (field.data_z != nullptr) cudaFree(field.data_z);
            field.data_x = nullptr;
            field.data_y = nullptr;
            field.data_z = nullptr;
        }
        context.device.occupancy_float = nullptr;
        context.device.occupancy       = nullptr;
        context.device.solid_temperature = nullptr;
    }

    void check_cuda(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return;
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    void update_pressure_anchor(ContextStorage& context) {
        if (context.step_runtime.occupancy_host.size() != context.cell_count) context.step_runtime.occupancy_host.resize(static_cast<std::size_t>(context.cell_count));
        check_cuda(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize occupancy");
        check_cuda(cudaMemcpy(context.step_runtime.occupancy_host.data(), context.device.occupancy, context.cell_count * sizeof(uint8_t), cudaMemcpyDeviceToHost), "cudaMemcpy occupancy_host");
        context.step_runtime.pressure_anchor = 0;
        for (std::uint64_t index = 0; index < context.cell_count; ++index) {
            if (context.step_runtime.occupancy_host[static_cast<std::size_t>(index)] == 0) {
                context.step_runtime.pressure_anchor = static_cast<int>(index);
                break;
            }
        }
    }

    void enforce_velocity_boundaries(ContextStorage& context) {
        const auto& solid_velocity_field = context.device.vector_fields[SMOKE_VECTOR_SOLID_VELOCITY];
        enforce_velocity_x_boundaries_kernel<<<context.velocity_x_cells, context.block, 0, context.stream>>>(context.device.flow.velocity_x, context.device.occupancy, solid_velocity_field.data_x, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        enforce_velocity_y_boundaries_kernel<<<context.velocity_y_cells, context.block, 0, context.stream>>>(context.device.flow.velocity_y, context.device.occupancy, solid_velocity_field.data_y, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        enforce_velocity_z_boundaries_kernel<<<context.velocity_z_cells, context.block, 0, context.stream>>>(context.device.flow.velocity_z, context.device.occupancy, solid_velocity_field.data_z, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        check_cuda(cudaGetLastError(), "enforce_velocity_boundaries_kernel");

        if (context.config.boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.ny + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.nz + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_velocity_x_kernel<<<grid, dim3(context.block.x, context.block.y, 1u), 0, context.stream>>>(context.device.flow.velocity_x, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_velocity_x_kernel");
        }
        if (context.config.boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.nx + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.nz + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_velocity_y_kernel<<<grid, dim3(context.block.x, context.block.y, 1u), 0, context.stream>>>(context.device.flow.velocity_y, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_velocity_y_kernel");
        }
        if (context.config.boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.nx + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.ny + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_velocity_z_kernel<<<grid, dim3(context.block.x, context.block.y, 1u), 0, context.stream>>>(context.device.flow.velocity_z, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_velocity_z_kernel");
        }
    }

    void solve_pressure(ContextStorage& context) {
        check_cuda(cudaMemsetAsync(context.device.flow.pressure, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pressure");
        check_cuda(cudaMemsetAsync(context.device.flow.pressure_rhs + context.step_runtime.pressure_anchor, 0, sizeof(float), context.stream), "cudaMemsetAsync pressure_rhs anchor");
        for (int iteration = 0; iteration < (std::max)(context.config.pressure_iterations, 1); ++iteration) {
            rbgs_pressure_kernel<<<context.cells, context.block, 0, context.stream>>>(context.device.flow.pressure, context.device.flow.pressure_rhs, context.device.occupancy, context.step_runtime.pressure_anchor, 0, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel red");
            rbgs_pressure_kernel<<<context.cells, context.block, 0, context.stream>>>(context.device.flow.pressure, context.device.flow.pressure_rhs, context.device.occupancy, context.step_runtime.pressure_anchor, 1, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel black");
        }
    }

} // namespace smoke_simulation

struct SmokeSimulationContext_t : smoke_simulation::ContextStorage {};

extern "C" {

SmokeSimulationResult smoke_simulation_create_context_cuda(const SmokeSimulationContextCreateDesc* desc, SmokeSimulationContext* out_context) {
    nvtx3::scoped_range range("smoke.create_context");
    if (desc == nullptr || out_context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if (desc->config.nx <= 0 || desc->config.ny <= 0 || desc->config.nz <= 0 || desc->config.cell_size <= 0.0f || desc->config.dt <= 0.0f) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    *out_context = nullptr;

    std::unique_ptr<SmokeSimulationContext_t> context{new (std::nothrow) SmokeSimulationContext_t{}};
    if (!context) return SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY;

    try {
        context->config          = desc->config;
        context->stream          = static_cast<cudaStream_t>(desc->stream);
        context->cell_count      = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
        context->velocity_x_count = static_cast<std::uint64_t>(context->config.nx + 1) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
        context->velocity_y_count = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny + 1) * static_cast<std::uint64_t>(context->config.nz);
        context->velocity_z_count = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz + 1);
        context->cell_bytes       = context->cell_count * sizeof(float);
        context->velocity_x_bytes = context->velocity_x_count * sizeof(float);
        context->velocity_y_bytes = context->velocity_y_count * sizeof(float);
        context->velocity_z_bytes = context->velocity_z_count * sizeof(float);
        if (context->stream == nullptr) {
            smoke_simulation::check_cuda(cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            context->owns_stream = true;
        }
        auto choose_block = [&]() {
            int min_grid_size = 0;
            int block_size    = 0;
            if (cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, smoke_simulation::advect_scalar_kernel, 0, 0) != cudaSuccess) return dim3(8u, 8u, 4u);
            if (block_size <= 0) return dim3(8u, 8u, 4u);
            unsigned block_z = block_size >= 256 ? 4u : block_size >= 128 ? 2u : 1u;
            unsigned block_y = block_size / static_cast<int>(block_z) >= 64 ? 8u : block_size / static_cast<int>(block_z) >= 32 ? 4u : 2u;
            unsigned block_x = static_cast<unsigned>((std::max)(block_size / static_cast<int>(block_y * block_z), 1));
            if (block_x > 16u) block_x = 16u;
            while (block_x * block_y * block_z > static_cast<unsigned>(block_size)) {
                if (block_x >= block_y && block_x > 1u) {
                    --block_x;
                    continue;
                }
                if (block_y >= block_z && block_y > 1u) {
                    --block_y;
                    continue;
                }
                if (block_z > 1u) {
                    --block_z;
                    continue;
                }
                break;
            }
            return dim3(block_x, block_y, block_z);
        };
        context->block            = choose_block();
        context->cells            = smoke_simulation::grid_for(context->config.nx, context->config.ny, context->config.nz, context->block);
        context->velocity_x_cells = smoke_simulation::grid_for(context->config.nx + 1, context->config.ny, context->config.nz, context->block);
        context->velocity_y_cells = smoke_simulation::grid_for(context->config.nx, context->config.ny + 1, context->config.nz, context->block);
        context->velocity_z_cells = smoke_simulation::grid_for(context->config.nx, context->config.ny, context->config.nz + 1, context->block);

        context->device.scalar_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::ScalarField{.kind = smoke_simulation::SMOKE_FIELD_DENSITY});
        context->device.scalar_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::ScalarField{.kind = smoke_simulation::SMOKE_FIELD_TEMPERATURE});
        context->device.vector_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::VectorField{.kind = smoke_simulation::SMOKE_VECTOR_FORCE});
        context->device.vector_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::VectorField{.kind = smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY});

        auto alloc_float = [](float** destination, const std::size_t bytes) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(destination), bytes), "cudaMalloc float");
        };
        auto alloc_u8 = [](uint8_t** destination, const std::size_t bytes) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(destination), bytes), "cudaMalloc uint8_t");
        };

        alloc_float(&context->device.flow.velocity_x, context->velocity_x_bytes);
        alloc_float(&context->device.flow.velocity_y, context->velocity_y_bytes);
        alloc_float(&context->device.flow.velocity_z, context->velocity_z_bytes);
        alloc_float(&context->device.flow.temp_velocity_x, context->velocity_x_bytes);
        alloc_float(&context->device.flow.temp_velocity_y, context->velocity_y_bytes);
        alloc_float(&context->device.flow.temp_velocity_z, context->velocity_z_bytes);
        alloc_float(&context->device.flow.centered_velocity_x, context->cell_bytes);
        alloc_float(&context->device.flow.centered_velocity_y, context->cell_bytes);
        alloc_float(&context->device.flow.centered_velocity_z, context->cell_bytes);
        alloc_float(&context->device.flow.velocity_magnitude, context->cell_bytes);
        alloc_float(&context->device.flow.pressure, context->cell_bytes);
        alloc_float(&context->device.flow.pressure_rhs, context->cell_bytes);
        alloc_float(&context->device.flow.divergence, context->cell_bytes);
        alloc_float(&context->device.flow.vorticity_x, context->cell_bytes);
        alloc_float(&context->device.flow.vorticity_y, context->cell_bytes);
        alloc_float(&context->device.flow.vorticity_z, context->cell_bytes);
        alloc_float(&context->device.flow.vorticity_magnitude, context->cell_bytes);
        alloc_float(&context->device.flow.force_x, context->cell_bytes);
        alloc_float(&context->device.flow.force_y, context->cell_bytes);
        alloc_float(&context->device.flow.force_z, context->cell_bytes);
        alloc_float(&context->device.occupancy_float, context->cell_bytes);
        alloc_u8(&context->device.occupancy, context->cell_count * sizeof(uint8_t));
        alloc_float(&context->device.solid_temperature, context->cell_bytes);
        for (auto& field : context->device.scalar_fields) {
            alloc_float(&field.data, context->cell_bytes);
            alloc_float(&field.temp, context->cell_bytes);
            alloc_float(&field.source, context->cell_bytes);
        }
        for (auto& field : context->device.vector_fields) {
            alloc_float(&field.data_x, context->cell_bytes);
            alloc_float(&field.data_y, context->cell_bytes);
            alloc_float(&field.data_z, context->cell_bytes);
        }

        const unsigned linear_grid = static_cast<unsigned>((context->cell_count + 255u) / 256u);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_x_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.velocity_x, 0.0f, context->velocity_x_count);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_y_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.velocity_y, 0.0f, context->velocity_y_count);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_z_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.velocity_z, 0.0f, context->velocity_z_count);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_x_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.temp_velocity_x, 0.0f, context->velocity_x_count);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_y_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.temp_velocity_y, 0.0f, context->velocity_y_count);
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((context->velocity_z_count + 255u) / 256u), 256, 0, context->stream>>>(context->device.flow.temp_velocity_z, 0.0f, context->velocity_z_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.centered_velocity_x, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.centered_velocity_y, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.centered_velocity_z, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.velocity_magnitude, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.pressure, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.pressure_rhs, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.divergence, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.vorticity_x, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.vorticity_y, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.vorticity_z, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.vorticity_magnitude, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.force_x, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.force_y, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.flow.force_z, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.occupancy_float, 0.0f, context->cell_count);
        smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(context->device.solid_temperature, context->config.ambient_temperature, context->cell_count);
        smoke_simulation::check_cuda(cudaMemsetAsync(context->device.occupancy, 0, context->cell_count * sizeof(uint8_t), context->stream), "cudaMemsetAsync occupancy");
        for (auto& field : context->device.scalar_fields) {
            const float initial_value = field.kind == smoke_simulation::SMOKE_FIELD_DENSITY ? desc->initial_density : desc->initial_temperature;
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data, initial_value, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.temp, initial_value, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.source, 0.0f, context->cell_count);
        }
        for (auto& field : context->device.vector_fields) {
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_x, 0.0f, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_y, 0.0f, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_z, 0.0f, context->cell_count);
        }
        smoke_simulation::check_cuda(cudaGetLastError(), "create_context init");

        *out_context = context.release();
        return SMOKE_SIMULATION_RESULT_OK;
    } catch (const std::bad_alloc&) {
        smoke_simulation::destroy_context_resources(*context);
        if (context->owns_stream && context->stream != nullptr) cudaStreamDestroy(context->stream);
        return SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY;
    } catch (...) {
        smoke_simulation::destroy_context_resources(*context);
        if (context->owns_stream && context->stream != nullptr) cudaStreamDestroy(context->stream);
        return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
}

SmokeSimulationResult smoke_simulation_destroy_context_cuda(SmokeSimulationContext context) {
    nvtx3::scoped_range range("smoke.destroy_context");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_OK;
    auto* storage = static_cast<smoke_simulation::ContextStorage*>(context);
    if (storage->stream != nullptr) cudaStreamSynchronize(storage->stream);
    smoke_simulation::destroy_context_resources(*storage);
    if (storage->owns_stream && storage->stream != nullptr) cudaStreamDestroy(storage->stream);
    delete storage;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_density_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_density");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage      = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& density_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY];
    if (values == nullptr) return cudaMemsetAsync(density_field.data, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(density_field.data, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_density_source_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_density_source");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage      = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& density_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY];
    if (values == nullptr) return cudaMemsetAsync(density_field.source, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(density_field.source, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_temperature_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_temperature");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage           = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& temperature_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE];
    if (values == nullptr) return cudaMemsetAsync(temperature_field.data, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(temperature_field.data, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_temperature_source_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_temperature_source");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage           = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& temperature_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE];
    if (values == nullptr) return cudaMemsetAsync(temperature_field.source, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(temperature_field.source, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_force_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z) {
    nvtx3::scoped_range range("smoke.update_force");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage      = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& force_field  = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE];
    if ((values_x == nullptr ? cudaMemsetAsync(force_field.data_x, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_x, values_x, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_y == nullptr ? cudaMemsetAsync(force_field.data_y, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_y, values_y, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_z == nullptr ? cudaMemsetAsync(force_field.data_z, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_z, values_z, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_occupancy_cuda(SmokeSimulationContext context, const uint8_t* values) {
    nvtx3::scoped_range range("smoke.update_occupancy");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);
    if (values == nullptr) {
        if (cudaMemsetAsync(storage.device.occupancy, 0, storage.cell_count * sizeof(uint8_t), storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (cudaMemsetAsync(storage.device.occupancy_float, 0, storage.cell_bytes, storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        storage.step_runtime.pressure_anchor = 0;
        storage.step_runtime.occupancy_dirty = false;
        return SMOKE_SIMULATION_RESULT_OK;
    }
    if (cudaMemcpyAsync(storage.device.occupancy, values, storage.cell_count * sizeof(uint8_t), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    smoke_simulation::copy_u8_to_float_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.occupancy_float, storage.device.occupancy, storage.cell_count);
    if (cudaGetLastError() != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    storage.step_runtime.occupancy_dirty = true;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_solid_velocity_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z) {
    nvtx3::scoped_range range("smoke.update_solid_velocity");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage              = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& solid_velocity_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY];
    if ((values_x == nullptr ? cudaMemsetAsync(solid_velocity_field.data_x, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_x, values_x, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_y == nullptr ? cudaMemsetAsync(solid_velocity_field.data_y, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_y, values_y, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_z == nullptr ? cudaMemsetAsync(solid_velocity_field.data_z, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_z, values_z, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_solid_temperature_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_solid_temperature");
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);
    if (values == nullptr) {
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.solid_temperature, storage.config.ambient_temperature, storage.cell_count);
        return cudaGetLastError() == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
    return cudaMemcpyAsync(storage.device.solid_temperature, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_step_cuda(SmokeSimulationContext context) {
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);

    try {
        nvtx3::scoped_range range("smoke.step");
        auto& density_field       = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY];
        auto& temperature_field   = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE];
        auto& force_field         = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE];

        {
            nvtx3::scoped_range bind_range("smoke.step.bind");
            if (storage.step_runtime.occupancy_dirty) {
                smoke_simulation::update_pressure_anchor(storage);
                storage.step_runtime.occupancy_dirty = false;
            }
        }

        {
            nvtx3::scoped_range velocity_force_range("smoke.step.velocity_forces");
            smoke_simulation::apply_solid_temperature_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(temperature_field.data, storage.device.occupancy, storage.device.solid_temperature, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature);
            smoke_simulation::check_cuda(cudaGetLastError(), "apply_solid_temperature_kernel pre");
            smoke_simulation::compute_center_velocity_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.centered_velocity_x, storage.device.flow.centered_velocity_y, storage.device.flow.centered_velocity_z, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_center_velocity_kernel");
            smoke_simulation::compute_vorticity_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.vorticity_x, storage.device.flow.vorticity_y, storage.device.flow.vorticity_z, storage.device.flow.vorticity_magnitude, storage.device.flow.centered_velocity_x, storage.device.flow.centered_velocity_y, storage.device.flow.centered_velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_vorticity_kernel");
            smoke_simulation::seed_force_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.flow.force_x, storage.device.flow.force_y, storage.device.flow.force_z, force_field.data_x, force_field.data_y, force_field.data_z, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "seed_force_kernel");
            smoke_simulation::add_buoyancy_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.force_y, density_field.data, temperature_field.data, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature, storage.config.buoyancy_density_factor, storage.config.buoyancy_temperature_factor, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "add_buoyancy_kernel");
            smoke_simulation::add_confinement_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.force_x, storage.device.flow.force_y, storage.device.flow.force_z, storage.device.flow.vorticity_x, storage.device.flow.vorticity_y, storage.device.flow.vorticity_z, storage.device.flow.vorticity_magnitude, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.vorticity_confinement, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "add_confinement_kernel");
            smoke_simulation::add_center_forces_to_velocity_x_kernel<<<storage.velocity_x_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_x, storage.device.flow.force_x, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
            smoke_simulation::add_center_forces_to_velocity_y_kernel<<<storage.velocity_y_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_y, storage.device.flow.force_y, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
            smoke_simulation::add_center_forces_to_velocity_z_kernel<<<storage.velocity_z_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_z, storage.device.flow.force_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
            smoke_simulation::check_cuda(cudaGetLastError(), "add_center_forces_to_velocity_kernel");
            smoke_simulation::enforce_velocity_boundaries(storage);
        }

        {
            nvtx3::scoped_range advect_velocity_range("smoke.step.advect_velocity");
            smoke_simulation::advect_velocity_x_kernel<<<storage.velocity_x_cells, storage.block, 0, storage.stream>>>(storage.device.flow.temp_velocity_x, storage.device.flow.velocity_x, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::advect_velocity_y_kernel<<<storage.velocity_y_cells, storage.block, 0, storage.stream>>>(storage.device.flow.temp_velocity_y, storage.device.flow.velocity_y, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::advect_velocity_z_kernel<<<storage.velocity_z_cells, storage.block, 0, storage.stream>>>(storage.device.flow.temp_velocity_z, storage.device.flow.velocity_z, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "advect_velocity_kernel");
            std::swap(storage.device.flow.velocity_x, storage.device.flow.temp_velocity_x);
            std::swap(storage.device.flow.velocity_y, storage.device.flow.temp_velocity_y);
            std::swap(storage.device.flow.velocity_z, storage.device.flow.temp_velocity_z);
            smoke_simulation::enforce_velocity_boundaries(storage);
        }

        {
            nvtx3::scoped_range project_range("smoke.step.project");
            smoke_simulation::compute_pressure_rhs_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.pressure_rhs, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.step_runtime.pressure_anchor, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_pressure_rhs_kernel");
            smoke_simulation::solve_pressure(storage);
            const auto& solid_velocity_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY];
            smoke_simulation::project_velocity_x_kernel<<<storage.velocity_x_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_x, storage.device.flow.pressure, storage.device.occupancy, solid_velocity_field.data_x, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::project_velocity_y_kernel<<<storage.velocity_y_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_y, storage.device.flow.pressure, storage.device.occupancy, solid_velocity_field.data_y, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::project_velocity_z_kernel<<<storage.velocity_z_cells, storage.block, 0, storage.stream>>>(storage.device.flow.velocity_z, storage.device.flow.pressure, storage.device.occupancy, solid_velocity_field.data_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "project_velocity_kernel");
            smoke_simulation::enforce_velocity_boundaries(storage);
        }

        {
            nvtx3::scoped_range temperature_range("smoke.step.update_temperature");
            smoke_simulation::add_source_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(temperature_field.temp, temperature_field.data, temperature_field.source, storage.config.dt, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "add_source_temperature_kernel");
            smoke_simulation::advect_scalar_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(temperature_field.data, temperature_field.temp, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.scalar_advection_mode, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "advect_temperature_kernel");
            smoke_simulation::apply_solid_temperature_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(temperature_field.data, storage.device.occupancy, storage.device.solid_temperature, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature);
            smoke_simulation::check_cuda(cudaGetLastError(), "apply_solid_temperature_kernel");
        }

        {
            nvtx3::scoped_range density_range("smoke.step.update_density");
            smoke_simulation::add_source_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(density_field.temp, density_field.data, density_field.source, storage.config.dt, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "add_source_density_kernel");
            smoke_simulation::advect_scalar_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(density_field.data, density_field.temp, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.scalar_advection_mode, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "advect_density_kernel");
            smoke_simulation::boundary_fill_density_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(density_field.temp, density_field.data, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "boundary_fill_density_kernel");
            std::swap(density_field.data, density_field.temp);
        }

        {
            nvtx3::scoped_range diagnostics_range("smoke.step.diagnostics");
            smoke_simulation::compute_center_velocity_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.centered_velocity_x, storage.device.flow.centered_velocity_y, storage.device.flow.centered_velocity_z, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.config.nx, storage.config.ny, storage.config.nz);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_center_velocity_kernel final");
            smoke_simulation::compute_vorticity_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.vorticity_x, storage.device.flow.vorticity_y, storage.device.flow.vorticity_z, storage.device.flow.vorticity_magnitude, storage.device.flow.centered_velocity_x, storage.device.flow.centered_velocity_y, storage.device.flow.centered_velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_vorticity_kernel final");
            smoke_simulation::compute_divergence_kernel<<<storage.cells, storage.block, 0, storage.stream>>>(storage.device.flow.divergence, storage.device.flow.velocity_x, storage.device.flow.velocity_y, storage.device.flow.velocity_z, storage.device.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_divergence_kernel");
            smoke_simulation::velocity_magnitude_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.flow.velocity_magnitude, storage.device.flow.centered_velocity_x, storage.device.flow.centered_velocity_y, storage.device.flow.centered_velocity_z, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "velocity_magnitude_kernel");
        }

        return SMOKE_SIMULATION_RESULT_OK;
    } catch (...) {
        return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
}

SmokeSimulationResult smoke_simulation_get_view_cuda(SmokeSimulationContext context, const SmokeSimulationViewRequest* request, SmokeSimulationView* out_view) {
    nvtx3::scoped_range range("smoke.get_view");
    if (context == nullptr || request == nullptr || out_view == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);
    *out_view     = SmokeSimulationView{
            .layout             = SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D,
            .nx                 = storage.config.nx,
            .ny                 = storage.config.ny,
            .nz                 = storage.config.nz,
            .row_stride_bytes   = static_cast<uint64_t>(storage.config.nx) * sizeof(float),
            .slice_stride_bytes = static_cast<uint64_t>(storage.config.nx) * static_cast<uint64_t>(storage.config.ny) * sizeof(float),
            .data0              = nullptr,
            .data1              = nullptr,
            .data2              = nullptr,
    };
    auto sync_consumer_stream = [&]() {
        if (request->consumer_stream == nullptr) return SMOKE_SIMULATION_RESULT_OK;
        auto consumer_stream = static_cast<cudaStream_t>(request->consumer_stream);
        if (consumer_stream == storage.stream) return SMOKE_SIMULATION_RESULT_OK;
        cudaEvent_t ready_event = nullptr;
        if (cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (cudaEventRecord(ready_event, storage.stream) != cudaSuccess) {
            cudaEventDestroy(ready_event);
            return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        }
        if (cudaStreamWaitEvent(consumer_stream, ready_event) != cudaSuccess) {
            cudaEventDestroy(ready_event);
            return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        }
        cudaEventDestroy(ready_event);
        return SMOKE_SIMULATION_RESULT_OK;
    };

    if (request->kind == SMOKE_SIMULATION_VIEW_DENSITY) {
        out_view->data0 = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_DENSITY_SOURCE) {
        out_view->data0 = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].source;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_TEMPERATURE) {
        out_view->data0 = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_TEMPERATURE_SOURCE) {
        out_view->data0 = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].source;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FORCE) {
        const auto& force_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE];
        out_view->layout        = SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0         = force_field.data_x;
        out_view->data1         = force_field.data_y;
        out_view->data2         = force_field.data_z;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_SOLID_VELOCITY) {
        const auto& solid_velocity_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY];
        out_view->layout                 = SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0                  = solid_velocity_field.data_x;
        out_view->data1                  = solid_velocity_field.data_y;
        out_view->data2                  = solid_velocity_field.data_z;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_SOLID_TEMPERATURE) {
        out_view->data0 = storage.device.solid_temperature;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_VELOCITY) {
        out_view->layout = SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0  = storage.device.flow.centered_velocity_x;
        out_view->data1  = storage.device.flow.centered_velocity_y;
        out_view->data2  = storage.device.flow.centered_velocity_z;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_VELOCITY_MAGNITUDE) {
        out_view->data0 = storage.device.flow.velocity_magnitude;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_PRESSURE) {
        out_view->data0 = storage.device.flow.pressure;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_PRESSURE_RHS) {
        out_view->data0 = storage.device.flow.pressure_rhs;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_DIVERGENCE) {
        out_view->data0 = storage.device.flow.divergence;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_VORTICITY) {
        out_view->layout = SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D_SOA3;
        out_view->data0  = storage.device.flow.vorticity_x;
        out_view->data1  = storage.device.flow.vorticity_y;
        out_view->data2  = storage.device.flow.vorticity_z;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_FLOW_VORTICITY_MAGNITUDE) {
        out_view->data0 = storage.device.flow.vorticity_magnitude;
        return sync_consumer_stream();
    }
    if (request->kind == SMOKE_SIMULATION_VIEW_OCCUPANCY) {
        out_view->data0 = storage.device.occupancy_float;
        return sync_consumer_stream();
    }
    return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

} // extern "C"
