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

    struct MacVelocityBuffers {
        float* u     = nullptr;
        float* v     = nullptr;
        float* w     = nullptr;
        float* u_tmp = nullptr;
        float* v_tmp = nullptr;
        float* w_tmp = nullptr;

        float* cell_x = nullptr;
        float* cell_y = nullptr;
        float* cell_z = nullptr;
    };

    struct ScalarBuffers {
        float* density         = nullptr;
        float* density_tmp     = nullptr;
        float* temperature     = nullptr;
        float* temperature_tmp = nullptr;
        float* pressure        = nullptr;
        float* pressure_rhs    = nullptr;
        float* divergence      = nullptr;
        float* vorticity_x     = nullptr;
        float* vorticity_y     = nullptr;
        float* vorticity_z     = nullptr;
        float* vorticity_mag   = nullptr;
        float* force_x         = nullptr;
        float* force_y         = nullptr;
        float* force_z         = nullptr;
        float* occupancy_float = nullptr;
        uint8_t* occupancy     = nullptr;
    };

    struct ContextStorage {
        SmokeSimulationConfig config{};
        cudaStream_t stream = nullptr;
        bool owns_stream    = false;

        dim3 block{};
        dim3 cell_grid{};
        dim3 u_grid{};
        dim3 v_grid{};
        dim3 w_grid{};

        std::uint64_t cell_count = 0;
        std::uint64_t u_count    = 0;
        std::uint64_t v_count    = 0;
        std::uint64_t w_count    = 0;

        std::size_t cell_bytes = 0;
        std::size_t u_bytes    = 0;
        std::size_t v_bytes    = 0;
        std::size_t w_bytes    = 0;

        MacVelocityBuffers velocity{};
        ScalarBuffers scalar{};
        int pressure_anchor = 0;
        std::vector<uint8_t> occupancy_host{};
    };

    void check_cuda(cudaError_t status, const char* what);

    __host__ __device__ inline std::uint64_t cell_index(const int x, const int y, const int z, const int nx, const int ny) {
        return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(x);
    }

    __host__ __device__ inline std::uint64_t u_index(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx + 1) + static_cast<std::uint64_t>(i);
    }

    __host__ __device__ inline std::uint64_t v_index(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
    }

    __host__ __device__ inline std::uint64_t w_index(const int i, const int j, const int k, const int nx, const int ny) {
        return static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j) * static_cast<std::uint64_t>(nx) + static_cast<std::uint64_t>(i);
    }

    inline dim3 grid_for(const int sx, const int sy, const int sz, const dim3 block) {
        return dim3(
            static_cast<unsigned>((sx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
            static_cast<unsigned>((sy + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
            static_cast<unsigned>((sz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
    }

    __host__ __device__ inline int wrap_index(int value, const int size) {
        if (size <= 0) return 0;
        value %= size;
        if (value < 0) value += size;
        return value;
    }

    __host__ __device__ inline bool cell_in_bounds(const int x, const int y, const int z, const int nx, const int ny, const int nz) {
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
        return occupancy[cell_index(x, y, z, nx, ny)] != 0;
    }

    __device__ float load_scalar(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        return field[cell_index(x, y, z, nx, ny)];
    }

    __device__ float load_u_face(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i > nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0.0f;
        return field[u_index(i, j, k, nx, ny)];
    }

    __device__ float load_v_face(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i >= nx || j < 0 || j > ny || k < 0 || k >= nz) return 0.0f;
        return field[v_index(i, j, k, nx, ny)];
    }

    __device__ float load_w_face(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nx > 0) i = wrap_index(i, nx);
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC && ny > 0) j = wrap_index(j, ny);
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC && nz > 0) k = wrap_index(k, nz);
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k > nz) return 0.0f;
        return field[w_index(i, j, k, nx, ny)];
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

    __device__ float sample_u_component(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
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

        const float c000 = load_u_face(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_u_face(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_u_face(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_u_face(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_u_face(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_u_face(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_u_face(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_u_face(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_v_component(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
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

        const float c000 = load_v_face(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_v_face(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_v_face(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_v_face(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_v_face(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_v_face(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_v_face(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_v_face(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float sample_w_component(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
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

        const float c000 = load_w_face(field, i0, j0, k0, nx, ny, nz, boundary);
        const float c100 = load_w_face(field, i1, j0, k0, nx, ny, nz, boundary);
        const float c010 = load_w_face(field, i0, j1, k0, nx, ny, nz, boundary);
        const float c110 = load_w_face(field, i1, j1, k0, nx, ny, nz, boundary);
        const float c001 = load_w_face(field, i0, j0, k1, nx, ny, nz, boundary);
        const float c101 = load_w_face(field, i1, j0, k1, nx, ny, nz, boundary);
        const float c011 = load_w_face(field, i0, j1, k1, nx, ny, nz, boundary);
        const float c111 = load_w_face(field, i1, j1, k1, nx, ny, nz, boundary);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0  = c00 + (c10 - c00) * ty;
        const float c1  = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

    __device__ float3 sample_mac_velocity(const float* u, const float* v, const float* w, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        return make_float3(
            sample_u_component(u, x, y, z, nx, ny, nz, h, boundary),
            sample_v_component(v, x, y, z, nx, ny, nz, h, boundary),
            sample_w_component(w, x, y, z, nx, ny, nz, h, boundary));
    }

    __device__ float3 trace_particle_rk2(const float3 start, const float* u, const float* v, const float* w, const uint8_t* occupancy, const float dt, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        const float3 vel0 = sample_mac_velocity(u, v, w, start.x, start.y, start.z, nx, ny, nz, h, boundary);
        const float3 mid  = make_float3(start.x - 0.5f * dt * vel0.x, start.y - 0.5f * dt * vel0.y, start.z - 0.5f * dt * vel0.z);
        const float3 vel1 = sample_mac_velocity(u, v, w, mid.x, mid.y, mid.z, nx, ny, nz, h, boundary);
        float3 traced     = make_float3(start.x - dt * vel1.x, start.y - dt * vel1.y, start.z - dt * vel1.z);
        float end_x       = traced.x;
        float end_y       = traced.y;
        float end_z       = traced.z;
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
            traced_hits_solid = !cell_in_bounds(end_cell_x, end_cell_y, end_cell_z, nx, ny, nz) || occupancy[cell_index(end_cell_x, end_cell_y, end_cell_z, nx, ny)] != 0;
        }
        if (!traced_hits_solid) return traced;

        float lo = 0.0f;
        float hi = 1.0f;
        for (int iter = 0; iter < 10; ++iter) {
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
                test_hits_solid = !cell_in_bounds(test_cell_x, test_cell_y, test_cell_z, nx, ny, nz) || occupancy[cell_index(test_cell_x, test_cell_y, test_cell_z, nx, ny)] != 0;
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

    __global__ void compute_center_velocity_kernel(float* cell_x, float* cell_y, float* cell_z, const float* u, const float* v, const float* w, const int nx, const int ny, const int nz) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = cell_index(x, y, z, nx, ny);
        cell_x[index]    = 0.5f * (u[u_index(x, y, z, nx, ny)] + u[u_index(x + 1, y, z, nx, ny)]);
        cell_y[index]    = 0.5f * (v[v_index(x, y, z, nx, ny)] + v[v_index(x, y + 1, z, nx, ny)]);
        cell_z[index]    = 0.5f * (w[w_index(x, y, z, nx, ny)] + w[w_index(x, y, z + 1, nx, ny)]);
    }

    __global__ void compute_vorticity_kernel(float* omega_x, float* omega_y, float* omega_z, float* omega_mag, const float* cell_x, const float* cell_y, const float* cell_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = cell_index(x, y, z, nx, ny);
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) {
            omega_x[index] = 0.0f;
            omega_y[index] = 0.0f;
            omega_z[index] = 0.0f;
            omega_mag[index] = 0.0f;
            return;
        }

        const float dvz_dy = 0.5f * (load_scalar(cell_z, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(cell_z, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float dvy_dz = 0.5f * (load_scalar(cell_y, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(cell_y, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float dux_dz = 0.5f * (load_scalar(cell_x, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(cell_x, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float duz_dx = 0.5f * (load_scalar(cell_z, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(cell_z, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float duy_dx = 0.5f * (load_scalar(cell_y, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(cell_y, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float dux_dy = 0.5f * (load_scalar(cell_x, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(cell_x, x, y - 1, z, nx, ny, nz, boundary)) / h;

        const float wx = dvz_dy - dvy_dz;
        const float wy = dux_dz - duz_dx;
        const float wz = duy_dx - dux_dy;

        omega_x[index]   = wx;
        omega_y[index]   = wy;
        omega_z[index]   = wz;
        omega_mag[index] = sqrtf(wx * wx + wy * wy + wz * wz);
    }

    __global__ void seed_force_kernel(float* force_x, float* force_y, float* force_z, const float* source_x, const float* source_y, const float* source_z, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        force_x[index] = source_x != nullptr ? source_x[index] : 0.0f;
        force_y[index] = source_y != nullptr ? source_y[index] : 0.0f;
        force_z[index] = source_z != nullptr ? source_z[index] : 0.0f;
    }

    __global__ void add_buoyancy_kernel(float* force_y, const float* density, const float* temperature, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;
        const auto index = cell_index(x, y, z, nx, ny);
        force_y[index] += -density_factor * density[index] + temperature_factor * (temperature[index] - ambient_temperature);
    }

    __global__ void add_confinement_kernel(float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_mag, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float epsilon, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;

        const float grad_x = 0.5f * (load_scalar(omega_mag, x + 1, y, z, nx, ny, nz, boundary) - load_scalar(omega_mag, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float grad_y = 0.5f * (load_scalar(omega_mag, x, y + 1, z, nx, ny, nz, boundary) - load_scalar(omega_mag, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float grad_z = 0.5f * (load_scalar(omega_mag, x, y, z + 1, nx, ny, nz, boundary) - load_scalar(omega_mag, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float grad_mag = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
        if (grad_mag < 1.0e-6f) return;

        const float inv_grad = 1.0f / grad_mag;
        const float nx_v     = grad_x * inv_grad;
        const float ny_v     = grad_y * inv_grad;
        const float nz_v     = grad_z * inv_grad;
        const auto index     = cell_index(x, y, z, nx, ny);

        const float wx = omega_x[index];
        const float wy = omega_y[index];
        const float wz = omega_z[index];
        const float confinement_x = epsilon * h * (ny_v * wz - nz_v * wy);
        const float confinement_y = epsilon * h * (nz_v * wx - nx_v * wz);
        const float confinement_z = epsilon * h * (nx_v * wy - ny_v * wx);

        force_x[index] += confinement_x;
        force_y[index] += confinement_y;
        force_z[index] += confinement_z;
    }

    __global__ void add_center_forces_to_u_kernel(float* u, const float* force_x, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        float sum = 0.0f;
        float weight = 0.0f;
        if (i > 0) {
            sum += force_x[cell_index(i - 1, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (i < nx) {
            sum += force_x[cell_index(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) u[u_index(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __global__ void add_center_forces_to_v_kernel(float* v, const float* force_y, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        float sum = 0.0f;
        float weight = 0.0f;
        if (j > 0) {
            sum += force_y[cell_index(i, j - 1, k, nx, ny)];
            weight += 1.0f;
        }
        if (j < ny) {
            sum += force_y[cell_index(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) v[v_index(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __global__ void add_center_forces_to_w_kernel(float* w, const float* force_z, const int nx, const int ny, const int nz, const float dt) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        float sum = 0.0f;
        float weight = 0.0f;
        if (k > 0) {
            sum += force_z[cell_index(i, j, k - 1, nx, ny)];
            weight += 1.0f;
        }
        if (k < nz) {
            sum += force_z[cell_index(i, j, k, nx, ny)];
            weight += 1.0f;
        }
        if (weight > 0.0f) w[w_index(i, j, k, nx, ny)] += dt * (sum / weight);
    }

    __device__ float solid_velocity_value(const float* solid_velocity, const uint8_t* occupancy, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        if (solid_velocity == nullptr || occupancy == nullptr) return 0.0f;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        if (occupancy[cell_index(x, y, z, nx, ny)] == 0) return 0.0f;
        return solid_velocity[cell_index(x, y, z, nx, ny)];
    }

    __global__ void enforce_u_boundaries_kernel(float* u, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = u[u_index(i, j, k, nx, ny)];
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
        const bool has_left  = resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary);
        const bool has_right = resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary);
        const bool left_occupied  = has_left && occupancy[cell_index(left_x, left_y, left_z, nx, ny)] != 0;
        const bool right_occupied = has_right && occupancy[cell_index(right_x, right_y, right_z, nx, ny)] != 0;
        if (!left_occupied && !right_occupied) return;

        float value = 0.0f;
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

    __global__ void enforce_v_boundaries_kernel(float* v, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = v[v_index(i, j, k, nx, ny)];
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_FIXED && (j == 0 || j == ny)) {
            face = 0.0f;
            return;
        }
        if (occupancy == nullptr) return;

        int down_x = i;
        int down_y = j - 1;
        int down_z = k;
        int up_x   = i;
        int up_y   = j;
        int up_z   = k;
        const bool has_down = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up   = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied = has_down && occupancy[cell_index(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied   = has_up && occupancy[cell_index(up_x, up_y, up_z, nx, ny)] != 0;
        if (!down_occupied && !up_occupied) return;

        float value = 0.0f;
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

    __global__ void enforce_w_boundaries_kernel(float* w, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = w[w_index(i, j, k, nx, ny)];
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_FIXED && (k == 0 || k == nz)) {
            face = 0.0f;
            return;
        }
        if (occupancy == nullptr) return;

        int back_x  = i;
        int back_y  = j;
        int back_z  = k - 1;
        int front_x = i;
        int front_y = j;
        int front_z = k;
        const bool has_back  = resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary);
        const bool has_front = resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary);
        const bool back_occupied  = has_back && occupancy[cell_index(back_x, back_y, back_z, nx, ny)] != 0;
        const bool front_occupied = has_front && occupancy[cell_index(front_x, front_y, front_z, nx, ny)] != 0;
        if (!back_occupied && !front_occupied) return;

        float value = 0.0f;
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

    __global__ void sync_periodic_u_kernel(float* u, const int nx, const int ny, const int nz) {
        const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (j >= ny || k >= nz) return;
        u[u_index(nx, j, k, nx, ny)] = u[u_index(0, j, k, nx, ny)];
    }

    __global__ void sync_periodic_v_kernel(float* v, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || k >= nz) return;
        v[v_index(i, ny, k, nx, ny)] = v[v_index(i, 0, k, nx, ny)];
    }

    __global__ void sync_periodic_w_kernel(float* w, const int nx, const int ny, const int nz) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        if (i >= nx || j >= ny) return;
        w[w_index(i, j, nz, nx, ny)] = w[w_index(i, j, 0, nx, ny)];
    }

    __global__ void advect_u_kernel(float* destination, const float* source, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;
        const float3 start = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, u, v, w, occupancy, dt, nx, ny, nz, h, boundary);
        destination[u_index(i, j, k, nx, ny)] = sample_u_component(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_v_kernel(float* destination, const float* source, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;
        const float3 start = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, u, v, w, occupancy, dt, nx, ny, nz, h, boundary);
        destination[v_index(i, j, k, nx, ny)] = sample_v_component(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_w_kernel(float* destination, const float* source, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;
        const float3 start = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
        const float3 traced = trace_particle_rk2(start, u, v, w, occupancy, dt, nx, ny, nz, h, boundary);
        destination[w_index(i, j, k, nx, ny)] = sample_w_component(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_scalar_kernel(float* destination, const float* source, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t advection_mode, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) {
            destination[cell_index(x, y, z, nx, ny)] = 0.0f;
            return;
        }
        const float3 start = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const float3 traced = trace_particle_rk2(start, u, v, w, occupancy, dt, nx, ny, nz, h, boundary);
        destination[cell_index(x, y, z, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_scalar_cubic(source, traced.x, traced.y, traced.z, nx, ny, nz, h, boundary)
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

        const auto index = cell_index(x, y, z, nx, ny);
        if (occupancy == nullptr || occupancy[index] == 0) {
            destination[index] = source[index];
            return;
        }

        float sum = 0.0f;
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
            int nx_cell = x + offset[0];
            int ny_cell = y + offset[1];
            int nz_cell = z + offset[2];
            if (!resolve_cell_coordinates(nx_cell, ny_cell, nz_cell, nx, ny, nz, boundary)) continue;
            const auto neighbor_index = cell_index(nx_cell, ny_cell, nz_cell, nx, ny);
            if (occupancy[neighbor_index] != 0) continue;
            sum += source[neighbor_index];
            weight += 1.0f;
        }
        destination[index] = weight > 0.0f ? sum / weight : 0.0f;
    }

    __global__ void compute_pressure_rhs_kernel(float* rhs, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int anchor_row, const int nx, const int ny, const int nz, const float h, const float dt) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = cell_index(x, y, z, nx, ny);
        if (static_cast<int>(index) == anchor_row) {
            rhs[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            rhs[index] = 0.0f;
            return;
        }
        const float div = (u[u_index(x + 1, y, z, nx, ny)] - u[u_index(x, y, z, nx, ny)] + v[v_index(x, y + 1, z, nx, ny)] - v[v_index(x, y, z, nx, ny)] + w[w_index(x, y, z + 1, nx, ny)] - w[w_index(x, y, z, nx, ny)]) / h;
        rhs[index] = -div / dt;
    }

    __global__ void compute_divergence_kernel(float* divergence, const float* u, const float* v, const float* w, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = cell_index(x, y, z, nx, ny);
        if (occupancy != nullptr && occupancy[index] != 0) {
            divergence[index] = 0.0f;
            return;
        }
        divergence[index] = (u[u_index(x + 1, y, z, nx, ny)] - u[u_index(x, y, z, nx, ny)] + v[v_index(x, y + 1, z, nx, ny)] - v[v_index(x, y, z, nx, ny)] + w[w_index(x, y, z + 1, nx, ny)] - w[w_index(x, y, z, nx, ny)]) / h;
    }

    __global__ void rbgs_pressure_kernel(float* pressure, const float* rhs, const uint8_t* occupancy, const int anchor_row, const int parity, const int nx, const int ny, const int nz, const float h, const SmokeSimulationBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (((x + y + z) & 1) != parity) return;

        const auto index = cell_index(x, y, z, nx, ny);
        if (static_cast<int>(index) == anchor_row) {
            pressure[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            pressure[index] = 0.0f;
            return;
        }

        float diagonal = 0.0f;
        float sum = 0.0f;
        const int offsets[6][3] = {
            {-1, 0, 0},
            {1, 0, 0},
            {0, -1, 0},
            {0, 1, 0},
            {0, 0, -1},
            {0, 0, 1},
        };
        for (const auto& offset : offsets) {
            int nx_cell = x + offset[0];
            int ny_cell = y + offset[1];
            int nz_cell = z + offset[2];
            if (!resolve_cell_coordinates(nx_cell, ny_cell, nz_cell, nx, ny, nz, boundary)) continue;
            const auto neighbor_index = cell_index(nx_cell, ny_cell, nz_cell, nx, ny);
            if (occupancy != nullptr && occupancy[neighbor_index] != 0) continue;
            diagonal += 1.0f;
            if (static_cast<int>(neighbor_index) == anchor_row) continue;
            sum += pressure[neighbor_index];
        }
        pressure[index] = diagonal > 0.0f ? (sum + rhs[index] * h * h) / diagonal : 0.0f;
    }

    __global__ void project_u_kernel(float* u, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = u[u_index(i, j, k, nx, ny)];
        if (boundary.x == SMOKE_SIMULATION_BOUNDARY_FIXED && (i == 0 || i == nx)) {
            face = 0.0f;
            return;
        }

        int left_x  = i - 1;
        int left_y  = j;
        int left_z  = k;
        int right_x = i;
        int right_y = j;
        int right_z = k;
        const bool has_left  = resolve_cell_coordinates(left_x, left_y, left_z, nx, ny, nz, boundary);
        const bool has_right = resolve_cell_coordinates(right_x, right_y, right_z, nx, ny, nz, boundary);
        const bool left_occupied  = has_left && occupancy != nullptr && occupancy[cell_index(left_x, left_y, left_z, nx, ny)] != 0;
        const bool right_occupied = has_right && occupancy != nullptr && occupancy[cell_index(right_x, right_y, right_z, nx, ny)] != 0;
        if (left_occupied || right_occupied) {
            float value = 0.0f;
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
            const float p_right = pressure[cell_index(right_x, right_y, right_z, nx, ny)];
            const float p_left  = pressure[cell_index(left_x, left_y, left_z, nx, ny)];
            face -= dt * (p_right - p_left) / h;
        }
    }

    __global__ void project_v_kernel(float* v, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = v[v_index(i, j, k, nx, ny)];
        if (boundary.y == SMOKE_SIMULATION_BOUNDARY_FIXED && (j == 0 || j == ny)) {
            face = 0.0f;
            return;
        }

        int down_x = i;
        int down_y = j - 1;
        int down_z = k;
        int up_x   = i;
        int up_y   = j;
        int up_z   = k;
        const bool has_down = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up   = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied = has_down && occupancy != nullptr && occupancy[cell_index(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied   = has_up && occupancy != nullptr && occupancy[cell_index(up_x, up_y, up_z, nx, ny)] != 0;
        if (down_occupied || up_occupied) {
            float value = 0.0f;
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
            const float p_up   = pressure[cell_index(up_x, up_y, up_z, nx, ny)];
            const float p_down = pressure[cell_index(down_x, down_y, down_z, nx, ny)];
            face -= dt * (p_up - p_down) / h;
        }
    }

    __global__ void project_w_kernel(float* w, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = w[w_index(i, j, k, nx, ny)];
        if (boundary.z == SMOKE_SIMULATION_BOUNDARY_FIXED && (k == 0 || k == nz)) {
            face = 0.0f;
            return;
        }

        int back_x  = i;
        int back_y  = j;
        int back_z  = k - 1;
        int front_x = i;
        int front_y = j;
        int front_z = k;
        const bool has_back  = resolve_cell_coordinates(back_x, back_y, back_z, nx, ny, nz, boundary);
        const bool has_front = resolve_cell_coordinates(front_x, front_y, front_z, nx, ny, nz, boundary);
        const bool back_occupied  = has_back && occupancy != nullptr && occupancy[cell_index(back_x, back_y, back_z, nx, ny)] != 0;
        const bool front_occupied = has_front && occupancy != nullptr && occupancy[cell_index(front_x, front_y, front_z, nx, ny)] != 0;
        if (back_occupied || front_occupied) {
            float value = 0.0f;
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
            const float p_front = pressure[cell_index(front_x, front_y, front_z, nx, ny)];
            const float p_back  = pressure[cell_index(back_x, back_y, back_z, nx, ny)];
            face -= dt * (p_front - p_back) / h;
        }
    }

    __global__ void compute_velocity_magnitude_kernel(float* destination, const float* cell_x, const float* cell_y, const float* cell_z, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        const float vx = cell_x[index];
        const float vy = cell_y[index];
        const float vz = cell_z[index];
        destination[index] = sqrtf(vx * vx + vy * vy + vz * vz);
    }

    __global__ void pack_velocity_kernel(float* destination, const float* cell_x, const float* cell_y, const float* cell_z, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        destination[index]            = cell_x[index];
        destination[count + index]    = cell_y[index];
        destination[count * 2u + index] = cell_z[index];
    }

    void destroy_buffers(ContextStorage& context) {
        auto free_ptr = [](auto* pointer) {
            if (pointer != nullptr) cudaFree(pointer);
        };

        free_ptr(context.velocity.u);
        free_ptr(context.velocity.v);
        free_ptr(context.velocity.w);
        free_ptr(context.velocity.u_tmp);
        free_ptr(context.velocity.v_tmp);
        free_ptr(context.velocity.w_tmp);
        free_ptr(context.velocity.cell_x);
        free_ptr(context.velocity.cell_y);
        free_ptr(context.velocity.cell_z);

        free_ptr(context.scalar.density);
        free_ptr(context.scalar.density_tmp);
        free_ptr(context.scalar.temperature);
        free_ptr(context.scalar.temperature_tmp);
        free_ptr(context.scalar.pressure);
        free_ptr(context.scalar.pressure_rhs);
        free_ptr(context.scalar.divergence);
        free_ptr(context.scalar.vorticity_x);
        free_ptr(context.scalar.vorticity_y);
        free_ptr(context.scalar.vorticity_z);
        free_ptr(context.scalar.vorticity_mag);
        free_ptr(context.scalar.force_x);
        free_ptr(context.scalar.force_y);
        free_ptr(context.scalar.force_z);
        free_ptr(context.scalar.occupancy_float);
        free_ptr(context.scalar.occupancy);
    }

    void destroy_context_storage(ContextStorage& context) {
        destroy_buffers(context);
        if (context.owns_stream && context.stream != nullptr) {
            cudaStreamDestroy(context.stream);
            context.stream      = nullptr;
            context.owns_stream = false;
        }
    }

    void check_cuda(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return;
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }

    void launch_fill(float* destination, const float value, const std::uint64_t count, cudaStream_t stream) {
        if (count == 0) return;
        const int block_size = 256;
        const int grid_size  = static_cast<int>((count + static_cast<std::uint64_t>(block_size) - 1) / static_cast<std::uint64_t>(block_size));
        fill_float_kernel<<<grid_size, block_size, 0, stream>>>(destination, value, count);
        check_cuda(cudaGetLastError(), "fill_float_kernel");
    }

    void update_pressure_anchor(ContextStorage& context) {
        if (context.occupancy_host.size() != context.cell_count) context.occupancy_host.resize(context.cell_count);
        check_cuda(cudaMemcpy(context.occupancy_host.data(), context.scalar.occupancy, context.cell_count * sizeof(uint8_t), cudaMemcpyDeviceToHost), "cudaMemcpy occupancy_host");
        context.pressure_anchor = 0;
        for (std::uint64_t index = 0; index < context.cell_count; ++index) {
            if (context.occupancy_host[static_cast<std::size_t>(index)] == 0) {
                context.pressure_anchor = static_cast<int>(index);
                return;
            }
        }
    }

    void enforce_velocity_boundaries(ContextStorage& context, const SmokeSimulationStepDesc* desc) {
        const auto solid_velocity_x = desc != nullptr ? desc->solid_velocity_x : nullptr;
        const auto solid_velocity_y = desc != nullptr ? desc->solid_velocity_y : nullptr;
        const auto solid_velocity_z = desc != nullptr ? desc->solid_velocity_z : nullptr;

        enforce_u_boundaries_kernel<<<context.u_grid, context.block, 0, context.stream>>>(context.velocity.u, context.scalar.occupancy, solid_velocity_x, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        enforce_v_boundaries_kernel<<<context.v_grid, context.block, 0, context.stream>>>(context.velocity.v, context.scalar.occupancy, solid_velocity_y, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        enforce_w_boundaries_kernel<<<context.w_grid, context.block, 0, context.stream>>>(context.velocity.w, context.scalar.occupancy, solid_velocity_z, context.config.nx, context.config.ny, context.config.nz, context.config.boundary);
        check_cuda(cudaGetLastError(), "enforce_velocity_boundaries_kernel");

        if (context.config.boundary.x == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.ny + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.nz + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_u_kernel<<<grid, dim3(context.block.x, context.block.y, 1), 0, context.stream>>>(context.velocity.u, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_u_kernel");
        }
        if (context.config.boundary.y == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.nx + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.nz + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_v_kernel<<<grid, dim3(context.block.x, context.block.y, 1), 0, context.stream>>>(context.velocity.v, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_v_kernel");
        }
        if (context.config.boundary.z == SMOKE_SIMULATION_BOUNDARY_PERIODIC) {
            const dim3 grid(
                static_cast<unsigned>((context.config.nx + static_cast<int>(context.block.x) - 1) / static_cast<int>(context.block.x)),
                static_cast<unsigned>((context.config.ny + static_cast<int>(context.block.y) - 1) / static_cast<int>(context.block.y)),
                1u);
            sync_periodic_w_kernel<<<grid, dim3(context.block.x, context.block.y, 1), 0, context.stream>>>(context.velocity.w, context.config.nx, context.config.ny, context.config.nz);
            check_cuda(cudaGetLastError(), "sync_periodic_w_kernel");
        }
    }

    void solve_pressure(ContextStorage& context) {
        check_cuda(cudaMemsetAsync(context.scalar.pressure_rhs + context.pressure_anchor, 0, sizeof(float), context.stream), "cudaMemsetAsync pressure_rhs anchor");
        for (int iteration = 0; iteration < std::max(context.config.pressure_iterations, 1); ++iteration) {
            rbgs_pressure_kernel<<<context.cell_grid, context.block, 0, context.stream>>>(context.scalar.pressure, context.scalar.pressure_rhs, context.scalar.occupancy, context.pressure_anchor, 0, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel red");
            rbgs_pressure_kernel<<<context.cell_grid, context.block, 0, context.stream>>>(context.scalar.pressure, context.scalar.pressure_rhs, context.scalar.occupancy, context.pressure_anchor, 1, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.boundary);
            check_cuda(cudaGetLastError(), "rbgs_pressure_kernel black");
        }
    }

} // namespace smoke_simulation

struct SmokeSimulationContext_t : smoke_simulation::ContextStorage {};

extern "C" {

SmokeSimulationResult smoke_simulation_create_context_cuda(const SmokeSimulationContextCreateDesc* desc, SmokeSimulationContext* out_context) {
    nvtx3::scoped_range range("smoke.create_context");
    if (out_context == nullptr || desc == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    *out_context = nullptr;

    std::unique_ptr<SmokeSimulationContext_t> context;
    try {
        context.reset(new (std::nothrow) SmokeSimulationContext_t{});
        if (!context) return SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY;

        context->config = desc->config;
        context->stream = static_cast<cudaStream_t>(desc->stream);
        if (context->stream == nullptr) {
            smoke_simulation::check_cuda(cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
            context->owns_stream = true;
        }
        context->block     = dim3(static_cast<unsigned>(std::max(context->config.block_x, 1)), static_cast<unsigned>(std::max(context->config.block_y, 1)), static_cast<unsigned>(std::max(context->config.block_z, 1)));
        context->cell_grid = smoke_simulation::grid_for(context->config.nx, context->config.ny, context->config.nz, context->block);
        context->u_grid    = smoke_simulation::grid_for(context->config.nx + 1, context->config.ny, context->config.nz, context->block);
        context->v_grid    = smoke_simulation::grid_for(context->config.nx, context->config.ny + 1, context->config.nz, context->block);
        context->w_grid    = smoke_simulation::grid_for(context->config.nx, context->config.ny, context->config.nz + 1, context->block);

        context->cell_count = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
        context->u_count    = static_cast<std::uint64_t>(context->config.nx + 1) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
        context->v_count    = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny + 1) * static_cast<std::uint64_t>(context->config.nz);
        context->w_count    = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz + 1);

        context->cell_bytes = context->cell_count * sizeof(float);
        context->u_bytes    = context->u_count * sizeof(float);
        context->v_bytes    = context->v_count * sizeof(float);
        context->w_bytes    = context->w_count * sizeof(float);

        auto alloc_float = [](float** destination, std::size_t bytes) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(destination), bytes), "cudaMalloc float");
        };
        auto alloc_u8 = [](uint8_t** destination, std::size_t bytes) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(destination), bytes), "cudaMalloc uint8");
        };

        alloc_float(&context->velocity.u, context->u_bytes);
        alloc_float(&context->velocity.v, context->v_bytes);
        alloc_float(&context->velocity.w, context->w_bytes);
        alloc_float(&context->velocity.u_tmp, context->u_bytes);
        alloc_float(&context->velocity.v_tmp, context->v_bytes);
        alloc_float(&context->velocity.w_tmp, context->w_bytes);
        alloc_float(&context->velocity.cell_x, context->cell_bytes);
        alloc_float(&context->velocity.cell_y, context->cell_bytes);
        alloc_float(&context->velocity.cell_z, context->cell_bytes);

        alloc_float(&context->scalar.density, context->cell_bytes);
        alloc_float(&context->scalar.density_tmp, context->cell_bytes);
        alloc_float(&context->scalar.temperature, context->cell_bytes);
        alloc_float(&context->scalar.temperature_tmp, context->cell_bytes);
        alloc_float(&context->scalar.pressure, context->cell_bytes);
        alloc_float(&context->scalar.pressure_rhs, context->cell_bytes);
        alloc_float(&context->scalar.divergence, context->cell_bytes);
        alloc_float(&context->scalar.vorticity_x, context->cell_bytes);
        alloc_float(&context->scalar.vorticity_y, context->cell_bytes);
        alloc_float(&context->scalar.vorticity_z, context->cell_bytes);
        alloc_float(&context->scalar.vorticity_mag, context->cell_bytes);
        alloc_float(&context->scalar.force_x, context->cell_bytes);
        alloc_float(&context->scalar.force_y, context->cell_bytes);
        alloc_float(&context->scalar.force_z, context->cell_bytes);
        alloc_float(&context->scalar.occupancy_float, context->cell_bytes);
        alloc_u8(&context->scalar.occupancy, context->cell_count * sizeof(uint8_t));

        smoke_simulation::launch_fill(context->velocity.u, 0.0f, context->u_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.v, 0.0f, context->v_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.w, 0.0f, context->w_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.u_tmp, 0.0f, context->u_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.v_tmp, 0.0f, context->v_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.w_tmp, 0.0f, context->w_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.cell_x, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.cell_y, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->velocity.cell_z, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.density, desc->initial_density, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.density_tmp, desc->initial_density, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.temperature, desc->initial_temperature, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.temperature_tmp, desc->initial_temperature, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.pressure, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.pressure_rhs, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.divergence, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.vorticity_x, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.vorticity_y, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.vorticity_z, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.vorticity_mag, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.force_x, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.force_y, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.force_z, 0.0f, context->cell_count, context->stream);
        smoke_simulation::launch_fill(context->scalar.occupancy_float, 0.0f, context->cell_count, context->stream);
        smoke_simulation::check_cuda(cudaMemsetAsync(context->scalar.occupancy, 0, context->cell_count * sizeof(uint8_t), context->stream), "cudaMemsetAsync occupancy");

        *out_context = context.release();
        return SMOKE_SIMULATION_RESULT_OK;
    } catch (...) {
        if (context) {
            smoke_simulation::destroy_context_storage(*context);
        }
        *out_context = nullptr;
        return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
}

SmokeSimulationResult smoke_simulation_destroy_context_cuda(SmokeSimulationContext context) {
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_OK;
    auto* storage = static_cast<smoke_simulation::ContextStorage*>(context);
    if (storage->stream != nullptr) cudaStreamSynchronize(storage->stream);
    smoke_simulation::destroy_context_storage(*storage);
    delete context;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_step_cuda(SmokeSimulationContext context, const SmokeSimulationStepDesc* desc) {
    if (context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);

    try {
        nvtx3::scoped_range range("smoke.step");
        if (desc != nullptr && desc->occupancy != nullptr) {
            smoke_simulation::check_cuda(cudaMemcpyAsync(storage.scalar.occupancy, desc->occupancy, storage.cell_count * sizeof(uint8_t), cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync occupancy");
            smoke_simulation::check_cuda(cudaStreamSynchronize(storage.stream), "cudaStreamSynchronize occupancy");
            smoke_simulation::update_pressure_anchor(storage);
        } else {
            smoke_simulation::check_cuda(cudaMemsetAsync(storage.scalar.occupancy, 0, storage.cell_count * sizeof(uint8_t), storage.stream), "cudaMemsetAsync occupancy");
            storage.pressure_anchor = 0;
        }

        smoke_simulation::apply_solid_temperature_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.temperature, storage.scalar.occupancy, desc != nullptr ? desc->solid_temperature : nullptr, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature);
        smoke_simulation::check_cuda(cudaGetLastError(), "apply_solid_temperature_kernel pre");

        smoke_simulation::compute_center_velocity_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.config.nx, storage.config.ny, storage.config.nz);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_center_velocity_kernel");

        smoke_simulation::compute_vorticity_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.vorticity_x, storage.scalar.vorticity_y, storage.scalar.vorticity_z, storage.scalar.vorticity_mag, storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_vorticity_kernel");

        smoke_simulation::seed_force_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.force_x, storage.scalar.force_y, storage.scalar.force_z, desc != nullptr ? desc->force_x : nullptr, desc != nullptr ? desc->force_y : nullptr, desc != nullptr ? desc->force_z : nullptr, storage.cell_count);
        smoke_simulation::check_cuda(cudaGetLastError(), "seed_force_kernel");

        smoke_simulation::add_buoyancy_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.force_y, storage.scalar.density, storage.scalar.temperature, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature, storage.config.buoyancy_density_factor, storage.config.buoyancy_temperature_factor, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "add_buoyancy_kernel");

        smoke_simulation::add_confinement_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.force_x, storage.scalar.force_y, storage.scalar.force_z, storage.scalar.vorticity_x, storage.scalar.vorticity_y, storage.scalar.vorticity_z, storage.scalar.vorticity_mag, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.vorticity_confinement, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "add_confinement_kernel");

        smoke_simulation::add_center_forces_to_u_kernel<<<storage.u_grid, storage.block, 0, storage.stream>>>(storage.velocity.u, storage.scalar.force_x, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
        smoke_simulation::add_center_forces_to_v_kernel<<<storage.v_grid, storage.block, 0, storage.stream>>>(storage.velocity.v, storage.scalar.force_y, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
        smoke_simulation::add_center_forces_to_w_kernel<<<storage.w_grid, storage.block, 0, storage.stream>>>(storage.velocity.w, storage.scalar.force_z, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.dt);
        smoke_simulation::check_cuda(cudaGetLastError(), "add_center_forces_to_faces_kernel");

        smoke_simulation::enforce_velocity_boundaries(storage, desc);

        smoke_simulation::advect_u_kernel<<<storage.u_grid, storage.block, 0, storage.stream>>>(storage.velocity.u_tmp, storage.velocity.u, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::advect_v_kernel<<<storage.v_grid, storage.block, 0, storage.stream>>>(storage.velocity.v_tmp, storage.velocity.v, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::advect_w_kernel<<<storage.w_grid, storage.block, 0, storage.stream>>>(storage.velocity.w_tmp, storage.velocity.w, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "advect_velocity_kernel");
        std::swap(storage.velocity.u, storage.velocity.u_tmp);
        std::swap(storage.velocity.v, storage.velocity.v_tmp);
        std::swap(storage.velocity.w, storage.velocity.w_tmp);

        smoke_simulation::enforce_velocity_boundaries(storage, desc);

        smoke_simulation::compute_pressure_rhs_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.pressure_rhs, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.pressure_anchor, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_pressure_rhs_kernel");
        smoke_simulation::solve_pressure(storage);

        smoke_simulation::project_u_kernel<<<storage.u_grid, storage.block, 0, storage.stream>>>(storage.velocity.u, storage.scalar.pressure, storage.scalar.occupancy, desc != nullptr ? desc->solid_velocity_x : nullptr, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::project_v_kernel<<<storage.v_grid, storage.block, 0, storage.stream>>>(storage.velocity.v, storage.scalar.pressure, storage.scalar.occupancy, desc != nullptr ? desc->solid_velocity_y : nullptr, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::project_w_kernel<<<storage.w_grid, storage.block, 0, storage.stream>>>(storage.velocity.w, storage.scalar.pressure, storage.scalar.occupancy, desc != nullptr ? desc->solid_velocity_z : nullptr, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "project_velocity_kernel");
        smoke_simulation::enforce_velocity_boundaries(storage, desc);

        smoke_simulation::add_source_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.temperature_tmp, storage.scalar.temperature, desc != nullptr ? desc->temperature_source : nullptr, storage.config.dt, storage.cell_count);
        smoke_simulation::check_cuda(cudaGetLastError(), "add_source_temperature_kernel");
        smoke_simulation::advect_scalar_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.temperature, storage.scalar.temperature_tmp, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.scalar_advection_mode, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "advect_temperature_kernel");
        smoke_simulation::apply_solid_temperature_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.temperature, storage.scalar.occupancy, desc != nullptr ? desc->solid_temperature : nullptr, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.ambient_temperature);
        smoke_simulation::check_cuda(cudaGetLastError(), "apply_solid_temperature_kernel");

        smoke_simulation::add_source_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.density_tmp, storage.scalar.density, desc != nullptr ? desc->density_source : nullptr, storage.config.dt, storage.cell_count);
        smoke_simulation::check_cuda(cudaGetLastError(), "add_source_density_kernel");
        smoke_simulation::advect_scalar_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.density, storage.scalar.density_tmp, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.dt, storage.config.scalar_advection_mode, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "advect_density_kernel");
        smoke_simulation::boundary_fill_density_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.density_tmp, storage.scalar.density, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "boundary_fill_density_kernel");
        std::swap(storage.scalar.density, storage.scalar.density_tmp);

        smoke_simulation::compute_center_velocity_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.config.nx, storage.config.ny, storage.config.nz);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_center_velocity_kernel final");
        smoke_simulation::compute_vorticity_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.vorticity_x, storage.scalar.vorticity_y, storage.scalar.vorticity_z, storage.scalar.vorticity_mag, storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size, storage.config.boundary);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_vorticity_kernel final");
        smoke_simulation::compute_divergence_kernel<<<storage.cell_grid, storage.block, 0, storage.stream>>>(storage.scalar.divergence, storage.velocity.u, storage.velocity.v, storage.velocity.w, storage.scalar.occupancy, storage.config.nx, storage.config.ny, storage.config.nz, storage.config.cell_size);
        smoke_simulation::check_cuda(cudaGetLastError(), "compute_divergence_kernel");
        smoke_simulation::copy_u8_to_float_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.scalar.occupancy_float, storage.scalar.occupancy, storage.cell_count);
        smoke_simulation::check_cuda(cudaGetLastError(), "copy_u8_to_float_kernel");

        return SMOKE_SIMULATION_RESULT_OK;
    } catch (...) {
        return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
}

SmokeSimulationResult smoke_simulation_export_cuda(SmokeSimulationContext context, const SmokeSimulationExportDesc* desc, void* destination) {
    if (context == nullptr || desc == nullptr || destination == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);

    try {
        switch (desc->kind) {
        case SMOKE_SIMULATION_EXPORT_DENSITY:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.density, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync density");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_TEMPERATURE:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.temperature, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync temperature");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_PRESSURE:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.pressure, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync pressure");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_DIVERGENCE:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.divergence, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync divergence");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_VELOCITY:
            smoke_simulation::pack_velocity_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(static_cast<float*>(destination), storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "pack_velocity_kernel");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_VELOCITY_MAGNITUDE:
            smoke_simulation::compute_velocity_magnitude_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(static_cast<float*>(destination), storage.velocity.cell_x, storage.velocity.cell_y, storage.velocity.cell_z, storage.cell_count);
            smoke_simulation::check_cuda(cudaGetLastError(), "compute_velocity_magnitude_kernel");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_VORTICITY_MAGNITUDE:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.vorticity_mag, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync vorticity_mag");
            return SMOKE_SIMULATION_RESULT_OK;
        case SMOKE_SIMULATION_EXPORT_OCCUPANCY:
            smoke_simulation::check_cuda(cudaMemcpyAsync(destination, storage.scalar.occupancy_float, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream), "cudaMemcpyAsync occupancy");
            return SMOKE_SIMULATION_RESULT_OK;
        default:
            return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        }
    } catch (...) {
        return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
}

} // extern "C"
