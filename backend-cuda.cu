#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace visual_smoke {
    using Stream = cudaStream_t;

    namespace {

        constexpr int boundary_x_min_face = 0;
        constexpr int boundary_x_max_face = 1;
        constexpr int boundary_y_min_face = 2;
        constexpr int boundary_y_max_face = 3;
        constexpr int boundary_z_min_face = 4;
        constexpr int boundary_z_max_face = 5;

        constexpr int max_levels = 16;

        struct GridLevel {
            int nx;
            int ny;
            int nz;
            float* solution;
            float* rhs;
        };

        struct GridHierarchy {
            int level_count;
            GridLevel levels[max_levels];
        };

        struct VCycleConfig {
            int cycles;
            int pre_smooth;
            int post_smooth;
            int coarse_smooth;
        };

        inline dim3 make_grid(const int nx, const int ny, const int nz, const dim3& block) {
            return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
        }

        __host__ __device__ uint32_t boundary_type(const uint32_t boundary_pack, const int face) {
            return (boundary_pack >> (face * 3)) & 0x7u;
        }

        uint32_t make_boundary_pack(const uint32_t x_min, const uint32_t x_max, const uint32_t y_min, const uint32_t y_max, const uint32_t z_min, const uint32_t z_max) {
            return (x_min << (boundary_x_min_face * 3)) | (x_max << (boundary_x_max_face * 3)) | (y_min << (boundary_y_min_face * 3)) | (y_max << (boundary_y_max_face * 3)) | (z_min << (boundary_z_min_face * 3)) | (z_max << (boundary_z_max_face * 3));
        }

        __host__ __device__ inline std::uint64_t index_3d(int x, int y, int z, int sx, int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
        }

        __device__ inline float fetch_clamped(const float* field, int x, int y, int z, int sx, int sy, int sz) {
            return field[index_3d(std::clamp(x, 0, sx - 1), std::clamp(y, 0, sy - 1), std::clamp(z, 0, sz - 1), sx, sy)];
        }

        __device__ inline float monotonic_cubic(float p0, float p1, float p2, float p3, float t) {
            const float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
            const float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
            const float a2 = -0.5f * p0 + 0.5f * p2;
            const float a3 = p1;
            return std::clamp(((a0 * t + a1) * t + a2) * t + a3, fminf(p1, p2), fmaxf(p1, p2));
        }

        __device__ float sample_grid(const float* field, float gx, float gy, float gz, int sx, int sy, int sz, bool cubic) {
            gx = std::clamp(gx, 0.0f, static_cast<float>(sx - 1));
            gy = std::clamp(gy, 0.0f, static_cast<float>(sy - 1));
            gz = std::clamp(gz, 0.0f, static_cast<float>(sz - 1));

            if (!cubic) {
                const int x0   = std::clamp(static_cast<int>(floorf(gx)), 0, sx - 1);
                const int y0   = std::clamp(static_cast<int>(floorf(gy)), 0, sy - 1);
                const int z0   = std::clamp(static_cast<int>(floorf(gz)), 0, sz - 1);
                const int x1   = min(x0 + 1, sx - 1);
                const int y1   = min(y0 + 1, sy - 1);
                const int z1   = min(z0 + 1, sz - 1);
                const float tx = gx - static_cast<float>(x0);
                const float ty = gy - static_cast<float>(y0);
                const float tz = gz - static_cast<float>(z0);

                const float c000 = field[index_3d(x0, y0, z0, sx, sy)];
                const float c100 = field[index_3d(x1, y0, z0, sx, sy)];
                const float c010 = field[index_3d(x0, y1, z0, sx, sy)];
                const float c110 = field[index_3d(x1, y1, z0, sx, sy)];
                const float c001 = field[index_3d(x0, y0, z1, sx, sy)];
                const float c101 = field[index_3d(x1, y0, z1, sx, sy)];
                const float c011 = field[index_3d(x0, y1, z1, sx, sy)];
                const float c111 = field[index_3d(x1, y1, z1, sx, sy)];

                const float c00 = c000 + (c100 - c000) * tx;
                const float c10 = c010 + (c110 - c010) * tx;
                const float c01 = c001 + (c101 - c001) * tx;
                const float c11 = c011 + (c111 - c011) * tx;
                const float c0  = c00 + (c10 - c00) * ty;
                const float c1  = c01 + (c11 - c01) * ty;
                return c0 + (c1 - c0) * tz;
            }

            const int ix   = static_cast<int>(floorf(gx));
            const int iy   = static_cast<int>(floorf(gy));
            const int iz   = static_cast<int>(floorf(gz));
            const float tx = gx - static_cast<float>(ix);
            const float ty = gy - static_cast<float>(iy);
            const float tz = gz - static_cast<float>(iz);

            float yz[4][4];
            for (int zz = 0; zz < 4; ++zz) {
                for (int yy = 0; yy < 4; ++yy) {
                    yz[zz][yy] = monotonic_cubic(fetch_clamped(field, ix - 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 0, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 2, iy + yy - 1, iz + zz - 1, sx, sy, sz), tx);
                }
            }

            float zline[4];
            for (int zz = 0; zz < 4; ++zz) {
                zline[zz] = monotonic_cubic(yz[zz][0], yz[zz][1], yz[zz][2], yz[zz][3], ty);
            }
            return monotonic_cubic(zline[0], zline[1], zline[2], zline[3], tz);
        }

        __device__ float sample_scalar(const float* field, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h - 0.5f, nx, ny, nz, cubic);
        }

        __device__ float sample_u(const float* field, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            return sample_grid(field, pos.x / h, pos.y / h - 0.5f, pos.z / h - 0.5f, nx + 1, ny, nz, cubic);
        }

        __device__ float sample_v(const float* field, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h, pos.z / h - 0.5f, nx, ny + 1, nz, cubic);
        }

        __device__ float sample_w(const float* field, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            return sample_grid(field, pos.x / h - 0.5f, pos.y / h - 0.5f, pos.z / h, nx, ny, nz + 1, cubic);
        }

        __device__ float3 clamp_domain(float3 pos, int nx, int ny, int nz, float h) {
            return make_float3(std::clamp(pos.x, 0.0f, static_cast<float>(nx) * h), std::clamp(pos.y, 0.0f, static_cast<float>(ny) * h), std::clamp(pos.z, 0.0f, static_cast<float>(nz) * h));
        }

        __device__ float3 sample_velocity(const float* u, const float* v, const float* w, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            pos = clamp_domain(pos, nx, ny, nz, h);
            return make_float3(sample_u(u, pos, nx, ny, nz, h, cubic), sample_v(v, pos, nx, ny, nz, h, cubic), sample_w(w, pos, nx, ny, nz, h, cubic));
        }

        __device__ float center_u(const float* u, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = std::clamp(i, 0, nx - 1);
            const int cj = std::clamp(j, 0, ny - 1);
            const int ck = std::clamp(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(u, ci, cj, ck, nx + 1, ny, nz) + fetch_clamped(u, ci + 1, cj, ck, nx + 1, ny, nz));
        }

        __device__ float center_v(const float* v, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = std::clamp(i, 0, nx - 1);
            const int cj = std::clamp(j, 0, ny - 1);
            const int ck = std::clamp(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(v, ci, cj, ck, nx, ny + 1, nz) + fetch_clamped(v, ci, cj + 1, ck, nx, ny + 1, nz));
        }

        __device__ float center_w(const float* w, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = std::clamp(i, 0, nx - 1);
            const int cj = std::clamp(j, 0, ny - 1);
            const int ck = std::clamp(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(w, ci, cj, ck, nx, ny, nz + 1) + fetch_clamped(w, ci, cj, ck + 1, nx, ny, nz + 1));
        }

        __global__ void compute_vorticity_kernel(const float* u, const float* v, const float* w, float* omega_x, float* omega_y, float* omega_z, float* omega_mag, int nx, int ny, int nz, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) return;

            const float dw_dy = (center_w(w, i, j + 1, k, nx, ny, nz) - center_w(w, i, j - 1, k, nx, ny, nz)) / (2.0f * h);
            const float dv_dz = (center_v(v, i, j, k + 1, nx, ny, nz) - center_v(v, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
            const float du_dz = (center_u(u, i, j, k + 1, nx, ny, nz) - center_u(u, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
            const float dw_dx = (center_w(w, i + 1, j, k, nx, ny, nz) - center_w(w, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
            const float dv_dx = (center_v(v, i + 1, j, k, nx, ny, nz) - center_v(v, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
            const float du_dy = (center_u(u, i, j + 1, k, nx, ny, nz) - center_u(u, i, j - 1, k, nx, ny, nz)) / (2.0f * h);

            const float wx            = dw_dy - dv_dz;
            const float wy            = du_dz - dw_dx;
            const float wz            = dv_dx - du_dy;
            const std::uint64_t index = index_3d(i, j, k, nx, ny);
            omega_x[index]            = wx;
            omega_y[index]            = wy;
            omega_z[index]            = wz;
            omega_mag[index]          = sqrtf(wx * wx + wy * wy + wz * wz);
        }

        __global__ void compute_confinement_kernel(const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_mag, float* force_x, float* force_y, float* force_z, int nx, int ny, int nz, float epsilon, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) {
                return;
            }

            const auto mag = [&](int x, int y, int z) { return fetch_clamped(omega_mag, x, y, z, nx, ny, nz); };

            const float gx            = (mag(i + 1, j, k) - mag(i - 1, j, k)) / (2.0f * h);
            const float gy            = (mag(i, j + 1, k) - mag(i, j - 1, k)) / (2.0f * h);
            const float gz            = (mag(i, j, k + 1) - mag(i, j, k - 1)) / (2.0f * h);
            const float inv_len       = rsqrtf(fmaxf(gx * gx + gy * gy + gz * gz, 1.0e-12f));
            const float nxv           = gx * inv_len;
            const float nyv           = gy * inv_len;
            const float nzv           = gz * inv_len;
            const std::uint64_t index = index_3d(i, j, k, nx, ny);
            force_x[index]            = epsilon * h * (nyv * omega_z[index] - nzv * omega_y[index]);
            force_y[index]            = epsilon * h * (nzv * omega_x[index] - nxv * omega_z[index]);
            force_z[index]            = epsilon * h * (nxv * omega_y[index] - nyv * omega_x[index]);
        }

        __global__ void apply_forces_kernel(float* u, float* v, float* w, const float* density, const float* temperature, const float* force_x, const float* force_y, const float* force_z, int nx, int ny, int nz, float ambient_temperature, float density_buoyancy, float temperature_buoyancy, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > 0 && i < nx && j < ny && k < nz) u[index_3d(i, j, k, nx + 1, ny)] += 0.5f * dt * (force_x[index_3d(i - 1, j, k, nx, ny)] + force_x[index_3d(i, j, k, nx, ny)]);
            if (i < nx && j > 0 && j < ny && k < nz) {
                const std::uint64_t below   = index_3d(i, j - 1, k, nx, ny);
                const std::uint64_t above   = index_3d(i, j, k, nx, ny);
                const float density_avg     = 0.5f * (density[below] + density[above]);
                const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
                const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
                const float buoyancy        = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
                v[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
            }
            if (i < nx && j < ny && k > 0 && k < nz) w[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
        }

        __global__ void advect_velocity_kernel(float* dst_u, float* dst_v, float* dst_w, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz) {
                if (i == 0 || i == nx)
                    dst_u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
                else {
                    const float3 pos                     = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                    const float3 vel                     = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_u[index_3d(i, j, k, nx + 1, ny)] = sample_u(src_u, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
            if (i < nx && j <= ny && k < nz) {
                if (j == 0 || j == ny)
                    dst_v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
                else {
                    const float3 pos                     = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
                    const float3 vel                     = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_v[index_3d(i, j, k, nx, ny + 1)] = sample_v(src_v, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
            if (i < nx && j < ny && k <= nz) {
                if (k == 0 || k == nz)
                    dst_w[index_3d(i, j, k, nx, ny)] = 0.0f;
                else {
                    const float3 pos                 = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
                    const float3 vel                 = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_w[index_3d(i, j, k, nx, ny)] = sample_w(src_w, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
        }

        __global__ void compute_poisson_rhs_kernel(float* rhs, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) return;
            rhs[index_3d(i, j, k, nx, ny)] = -(fetch_clamped(u, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(u, i, j, k, nx + 1, ny, nz) + fetch_clamped(v, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(v, i, j, k, nx, ny + 1, nz) + fetch_clamped(w, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(w, i, j, k, nx, ny, nz + 1)) * (h / dt);
        }

        __global__ void poisson_rbgs_kernel(float* pressure, const float* rhs, int nx, int ny, int nz, int parity) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz || ((i + j + k) & 1) != parity) return;

            float sum = 0.0f;
            int count = 0;
            if (i > 0) {
                sum += pressure[index_3d(i - 1, j, k, nx, ny)];
                ++count;
            }
            if (i + 1 < nx) {
                sum += pressure[index_3d(i + 1, j, k, nx, ny)];
                ++count;
            }
            if (j > 0) {
                sum += pressure[index_3d(i, j - 1, k, nx, ny)];
                ++count;
            }
            if (j + 1 < ny) {
                sum += pressure[index_3d(i, j + 1, k, nx, ny)];
                ++count;
            }
            if (k > 0) {
                sum += pressure[index_3d(i, j, k - 1, nx, ny)];
                ++count;
            }
            if (k + 1 < nz) {
                sum += pressure[index_3d(i, j, k + 1, nx, ny)];
                ++count;
            }

            pressure[index_3d(i, j, k, nx, ny)] = count > 0 ? (sum + rhs[index_3d(i, j, k, nx, ny)]) / static_cast<float>(count) : 0.0f;
        }

        __global__ void restrict_poisson_residual_kernel(float* coarse_rhs, const float* fine_pressure, const float* fine_rhs, int fine_nx, int fine_ny, int fine_nz) {
            const int x         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y         = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z         = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
            const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
            const int coarse_nz = std::max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            float residual_sum = 0.0f;
            int samples        = 0;
            for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                        float neighbors = 0.0f;
                        int count       = 0;
                        if (fx > 0) {
                            neighbors += fine_pressure[index_3d(fx - 1, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        }
                        if (fx + 1 < fine_nx) {
                            neighbors += fine_pressure[index_3d(fx + 1, fy, fz, fine_nx, fine_ny)];
                            ++count;
                        }
                        if (fy > 0) {
                            neighbors += fine_pressure[index_3d(fx, fy - 1, fz, fine_nx, fine_ny)];
                            ++count;
                        }
                        if (fy + 1 < fine_ny) {
                            neighbors += fine_pressure[index_3d(fx, fy + 1, fz, fine_nx, fine_ny)];
                            ++count;
                        }
                        if (fz > 0) {
                            neighbors += fine_pressure[index_3d(fx, fy, fz - 1, fine_nx, fine_ny)];
                            ++count;
                        }
                        if (fz + 1 < fine_nz) {
                            neighbors += fine_pressure[index_3d(fx, fy, fz + 1, fine_nx, fine_ny)];
                            ++count;
                        }
                        residual_sum += fine_rhs[index_3d(fx, fy, fz, fine_nx, fine_ny)] - (static_cast<float>(count) * fine_pressure[index_3d(fx, fy, fz, fine_nx, fine_ny)] - neighbors);
                        ++samples;
                    }
                }
            }
            coarse_rhs[index_3d(x, y, z, coarse_nx, coarse_ny)] = samples > 0 ? residual_sum / static_cast<float>(samples) : 0.0f;
        }

        __global__ void prolongate_add_kernel(float* fine_pressure, const float* coarse_pressure, int fine_nx, int fine_ny, int fine_nz, int coarse_nx, int coarse_ny, int coarse_nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= fine_nx || j >= fine_ny || k >= fine_nz) return;
            fine_pressure[index_3d(i, j, k, fine_nx, fine_ny)] += sample_grid(coarse_pressure, 0.5f * static_cast<float>(i) - 0.25f, 0.5f * static_cast<float>(j) - 0.25f, 0.5f * static_cast<float>(k) - 0.25f, coarse_nx, coarse_ny, coarse_nz, false);
        }

        __global__ void project_velocity_kernel(float* u, float* v, float* w, const float* pressure, int nx, int ny, int nz, float scale) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz) {
                if (i == 0 || i == nx)
                    u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
                else
                    u[index_3d(i, j, k, nx + 1, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) * scale;
            }
            if (i < nx && j <= ny && k < nz) {
                if (j == 0 || j == ny)
                    v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
                else
                    v[index_3d(i, j, k, nx, ny + 1)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) * scale;
            }
            if (i < nx && j < ny && k <= nz) {
                if (k == 0 || k == nz)
                    w[index_3d(i, j, k, nx, ny)] = 0.0f;
                else
                    w[index_3d(i, j, k, nx, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) * scale;
            }
        }


        __global__ void advect_scalar_kernel(float* scalar_dst, const float* scalar_src, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h, float dt, bool cubic, int clamp_non_negative) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) return;
            const float3 pos                      = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
            const float3 vel                      = sample_velocity(u, v, w, pos, nx, ny, nz, h, cubic);
            const float3 back                     = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
            const float value                     = sample_scalar(scalar_src, back, nx, ny, nz, h, cubic);
            scalar_dst[index_3d(i, j, k, nx, ny)] = clamp_non_negative != 0 ? fmaxf(0.0f, value) : value;
        }

        __global__ void enforce_velocity_boundaries_kernel(
            float* velocity_x,
            float* velocity_y,
            float* velocity_z,
            const int nx,
            const int ny,
            const int nz,
            const uint32_t boundary_pack,
            const float inflow_velocity_x_min,
            const float inflow_velocity_x_max,
            const float inflow_velocity_y_min,
            const float inflow_velocity_y_max,
            const float inflow_velocity_z_min,
            const float inflow_velocity_z_max) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

            const uint32_t bx_min = boundary_type(boundary_pack, boundary_x_min_face);
            const uint32_t bx_max = boundary_type(boundary_pack, boundary_x_max_face);
            const uint32_t by_min = boundary_type(boundary_pack, boundary_y_min_face);
            const uint32_t by_max = boundary_type(boundary_pack, boundary_y_max_face);
            const uint32_t bz_min = boundary_type(boundary_pack, boundary_z_min_face);
            const uint32_t bz_max = boundary_type(boundary_pack, boundary_z_max_face);

            if (x <= nx && y < ny && z < nz) {
                const auto u_index = index_3d(x, y, z, nx + 1, ny);
                const bool touches_no_slip_tangent = (y == 0 && by_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (y == ny - 1 && by_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (z == 0 && bz_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (z == nz - 1 && bz_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP);
                if (touches_no_slip_tangent) velocity_x[u_index] = 0.0f;
                if (x == 0) {
                    if (bx_min == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_x[u_index] = inflow_velocity_x_min;
                    else if (bx_min == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_x[u_index] = velocity_x[index_3d(1, y, z, nx + 1, ny)];
                    else velocity_x[u_index] = 0.0f;
                }
                if (x == nx) {
                    if (bx_max == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_x[u_index] = inflow_velocity_x_max;
                    else if (bx_max == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_x[u_index] = velocity_x[index_3d(nx - 1, y, z, nx + 1, ny)];
                    else velocity_x[u_index] = 0.0f;
                }
            }

            if (x < nx && y <= ny && z < nz) {
                const auto v_index = index_3d(x, y, z, nx, ny + 1);
                const bool touches_no_slip_tangent = (x == 0 && bx_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (x == nx - 1 && bx_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (z == 0 && bz_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (z == nz - 1 && bz_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP);
                if (touches_no_slip_tangent) velocity_y[v_index] = 0.0f;
                if (y == 0) {
                    if (by_min == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_y[v_index] = inflow_velocity_y_min;
                    else if (by_min == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_y[v_index] = velocity_y[index_3d(x, 1, z, nx, ny + 1)];
                    else velocity_y[v_index] = 0.0f;
                }
                if (y == ny) {
                    if (by_max == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_y[v_index] = inflow_velocity_y_max;
                    else if (by_max == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_y[v_index] = velocity_y[index_3d(x, ny - 1, z, nx, ny + 1)];
                    else velocity_y[v_index] = 0.0f;
                }
            }

            if (x < nx && y < ny && z <= nz) {
                const auto w_index = index_3d(x, y, z, nx, ny);
                const bool touches_no_slip_tangent = (x == 0 && bx_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (x == nx - 1 && bx_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (y == 0 && by_min == VISUAL_SMOKE_BOUNDARY_NO_SLIP) || (y == ny - 1 && by_max == VISUAL_SMOKE_BOUNDARY_NO_SLIP);
                if (touches_no_slip_tangent) velocity_z[w_index] = 0.0f;
                if (z == 0) {
                    if (bz_min == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_z[w_index] = inflow_velocity_z_min;
                    else if (bz_min == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_z[w_index] = velocity_z[index_3d(x, y, 1, nx, ny)];
                    else velocity_z[w_index] = 0.0f;
                }
                if (z == nz) {
                    if (bz_max == VISUAL_SMOKE_BOUNDARY_INFLOW) velocity_z[w_index] = inflow_velocity_z_max;
                    else if (bz_max == VISUAL_SMOKE_BOUNDARY_OUTFLOW) velocity_z[w_index] = velocity_z[index_3d(x, y, nz - 1, nx, ny)];
                    else velocity_z[w_index] = 0.0f;
                }
            }
        }

        __global__ void apply_scalar_inflow_boundaries_kernel(
            float* scalar,
            const int nx,
            const int ny,
            const int nz,
            const uint32_t boundary_pack,
            const float inflow_scalar_x_min,
            const float inflow_scalar_x_max,
            const float inflow_scalar_y_min,
            const float inflow_scalar_y_max,
            const float inflow_scalar_z_min,
            const float inflow_scalar_z_max) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= nx || y >= ny || z >= nz) return;

            float sum = 0.0f;
            int count = 0;
            if (x == 0 && boundary_type(boundary_pack, boundary_x_min_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_x_min;
                ++count;
            }
            if (x == nx - 1 && boundary_type(boundary_pack, boundary_x_max_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_x_max;
                ++count;
            }
            if (y == 0 && boundary_type(boundary_pack, boundary_y_min_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_y_min;
                ++count;
            }
            if (y == ny - 1 && boundary_type(boundary_pack, boundary_y_max_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_y_max;
                ++count;
            }
            if (z == 0 && boundary_type(boundary_pack, boundary_z_min_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_z_min;
                ++count;
            }
            if (z == nz - 1 && boundary_type(boundary_pack, boundary_z_max_face) == VISUAL_SMOKE_BOUNDARY_INFLOW) {
                sum += inflow_scalar_z_max;
                ++count;
            }
            if (count > 0) scalar[index_3d(x, y, z, nx, ny)] = sum / static_cast<float>(count);
        }

        __global__ void add_scalar_source_kernel(float* destination, int sx, int sy, int sz, float center_x, float center_y, float center_z, float radius, float amount, float sample_offset_x, float sample_offset_y, float sample_offset_z) {
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (x >= sx || y >= sy || z >= sz) return;

            const float px      = static_cast<float>(x) + sample_offset_x;
            const float py      = static_cast<float>(y) + sample_offset_y;
            const float pz      = static_cast<float>(z) + sample_offset_z;
            const float dx      = px - center_x;
            const float dy      = py - center_y;
            const float dz      = pz - center_z;
            const float radius2 = radius * radius;
            const float dist2   = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2) return;
            destination[index_3d(x, y, z, sx, sy)] += amount * fmaxf(0.0f, 1.0f - dist2 / radius2);
        }

        GridHierarchy build_hierarchy(const int base_nx, const int base_ny, const int base_nz, float* base_solution, float* base_rhs, float* coarse_solution_storage, float* coarse_rhs_storage) {
            GridHierarchy hierarchy{.level_count = 1};
            hierarchy.levels[0] = GridLevel{
                .nx       = base_nx,
                .ny       = base_ny,
                .nz       = base_nz,
                .solution = base_solution,
                .rhs      = base_rhs,
            };

            std::uint64_t offset = 0;
            while (hierarchy.level_count < max_levels) {
                const GridLevel& previous = hierarchy.levels[hierarchy.level_count - 1];
                if (previous.nx <= 1 && previous.ny <= 1 && previous.nz <= 1) break;

                const int nx = std::max(1, (previous.nx + 1) / 2);
                const int ny = std::max(1, (previous.ny + 1) / 2);
                const int nz = std::max(1, (previous.nz + 1) / 2);

                hierarchy.levels[hierarchy.level_count] = GridLevel{
                    .nx       = nx,
                    .ny       = ny,
                    .nz       = nz,
                    .solution = coarse_solution_storage + offset,
                    .rhs      = coarse_rhs_storage + offset,
                };

                offset += static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
                ++hierarchy.level_count;
            }

            return hierarchy;
        }

        struct PoissonVCycleOps {
            int clear_coarse_solution(const GridLevel& level, const Stream stream) const {
                const auto bytes = static_cast<std::uint64_t>(level.nx) * static_cast<std::uint64_t>(level.ny) * static_cast<std::uint64_t>(level.nz) * sizeof(float);
                return cudaMemsetAsync(level.solution, 0, bytes, stream) == cudaSuccess ? 0 : 5001;
            }

            int smooth_level(const GridLevel& level, const dim3& block, const Stream stream, const int iterations) const {
                const dim3 grid = make_grid(level.nx, level.ny, level.nz, block);
                for (int i = 0; i < iterations; ++i) {
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 0);
                    poisson_rbgs_kernel<<<grid, block, 0, stream>>>(level.solution, level.rhs, level.nx, level.ny, level.nz, 1);
                }
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int restrict_residual(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                restrict_poisson_residual_kernel<<<make_grid(coarse.nx, coarse.ny, coarse.nz, block), block, 0, stream>>>(coarse.rhs, fine.solution, fine.rhs, fine.nx, fine.ny, fine.nz);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }

            int prolongate_and_postprocess(const GridLevel& fine, const GridLevel& coarse, const dim3& block, const Stream stream) const {
                prolongate_add_kernel<<<make_grid(fine.nx, fine.ny, fine.nz, block), block, 0, stream>>>(fine.solution, coarse.solution, fine.nx, fine.ny, fine.nz, coarse.nx, coarse.ny, coarse.nz);
                return cudaGetLastError() == cudaSuccess ? 0 : 5001;
            }
        };

        template <class TOps>
        int run_v_cycle(const GridHierarchy& hierarchy, const VCycleConfig& config, const TOps& ops, const dim3& block, const Stream stream) {
            for (int cycle = 0; cycle < config.cycles; ++cycle) {
                for (int level_index = 0; level_index + 1 < hierarchy.level_count; ++level_index) {
                    const GridLevel& fine   = hierarchy.levels[level_index];
                    const GridLevel& coarse = hierarchy.levels[level_index + 1];

                    if (const int code = ops.smooth_level(fine, block, stream, config.pre_smooth); code != 0) return code;
                    if (const int code = ops.clear_coarse_solution(coarse, stream); code != 0) return code;
                    if (const int code = ops.restrict_residual(fine, coarse, block, stream); code != 0) return code;
                }

                if (const int code = ops.smooth_level(hierarchy.levels[hierarchy.level_count - 1], block, stream, config.coarse_smooth); code != 0) return code;

                for (int level_index = hierarchy.level_count - 2; level_index >= 0; --level_index) {
                    const GridLevel& fine   = hierarchy.levels[level_index];
                    const GridLevel& coarse = hierarchy.levels[level_index + 1];

                    if (const int code = ops.prolongate_and_postprocess(fine, coarse, block, stream); code != 0) return code;
                    if (const int code = ops.smooth_level(fine, block, stream, config.post_smooth); code != 0) return code;
                }
            }

            return 0;
        }

    } // namespace

} // namespace visual_smoke

extern "C" {

int32_t visual_simulation_of_smoke_forces_cuda(const VisualSimulationOfSmokeForcesDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_forces_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells         = make_grid(desc->nx, desc->ny, desc->nz, block);
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const uint32_t boundary_pack = make_boundary_pack(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max);
    const auto stream        = static_cast<Stream>(desc->stream);

    nvtx3::scoped_range range("vsmoke.step.forces");
    compute_vorticity_kernel<<<cells, block, 0, stream>>>(static_cast<float*>(desc->velocity_x), static_cast<float*>(desc->velocity_y), static_cast<float*>(desc->velocity_z), static_cast<float*>(desc->temporary_omega_x), static_cast<float*>(desc->temporary_omega_y), static_cast<float*>(desc->temporary_omega_z),
        static_cast<float*>(desc->temporary_omega_magnitude), desc->nx, desc->ny, desc->nz, desc->cell_size);
    compute_confinement_kernel<<<cells, block, 0, stream>>>(static_cast<float*>(desc->temporary_omega_x), static_cast<float*>(desc->temporary_omega_y), static_cast<float*>(desc->temporary_omega_z), static_cast<float*>(desc->temporary_omega_magnitude), static_cast<float*>(desc->temporary_force_x), static_cast<float*>(desc->temporary_force_y),
        static_cast<float*>(desc->temporary_force_z), desc->nx, desc->ny, desc->nz, desc->vorticity_epsilon, desc->cell_size);
    apply_forces_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->velocity_x), static_cast<float*>(desc->velocity_y), static_cast<float*>(desc->velocity_z), static_cast<float*>(desc->density), static_cast<float*>(desc->temperature), static_cast<float*>(desc->temporary_force_x), static_cast<float*>(desc->temporary_force_y),
        static_cast<float*>(desc->temporary_force_z), desc->nx, desc->ny, desc->nz, desc->ambient_temperature, desc->density_buoyancy, desc->temperature_buoyancy, desc->dt);
    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->velocity_x), static_cast<float*>(desc->velocity_y), static_cast<float*>(desc->velocity_z), desc->nx, desc->ny, desc->nz, boundary_pack, desc->inflow_velocity_x_min, desc->inflow_velocity_x_max, desc->inflow_velocity_y_min,
        desc->inflow_velocity_y_max, desc->inflow_velocity_z_min, desc->inflow_velocity_z_max);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t visual_simulation_of_smoke_advect_velocity_cuda(const VisualSimulationOfSmokeAdvectVelocityDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_advect_velocity_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 velocity_grid = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const bool cubic         = desc->use_monotonic_cubic != 0u;
    const uint32_t boundary_pack = make_boundary_pack(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max);
    const auto stream        = static_cast<Stream>(desc->stream);

    nvtx3::scoped_range range("vsmoke.step.advect_velocity");
    advect_velocity_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->temporary_previous_velocity_x), static_cast<float*>(desc->temporary_previous_velocity_y), static_cast<float*>(desc->temporary_previous_velocity_z), static_cast<float*>(desc->velocity_x), static_cast<float*>(desc->velocity_y), static_cast<float*>(desc->velocity_z),
        desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, cubic);
    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->temporary_previous_velocity_x), static_cast<float*>(desc->temporary_previous_velocity_y), static_cast<float*>(desc->temporary_previous_velocity_z), desc->nx, desc->ny, desc->nz, boundary_pack, desc->inflow_velocity_x_min,
        desc->inflow_velocity_x_max, desc->inflow_velocity_y_min, desc->inflow_velocity_y_max, desc->inflow_velocity_z_min, desc->inflow_velocity_z_max);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}

int32_t visual_simulation_of_smoke_project_cuda(const VisualSimulationOfSmokeProjectDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_project_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells                       = make_grid(desc->nx, desc->ny, desc->nz, block);
    const dim3 velocity_grid               = make_grid(desc->nx + 1, desc->ny + 1, desc->nz + 1, block);
    const uint32_t boundary_pack           = make_boundary_pack(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max);
    const auto stream                      = static_cast<Stream>(desc->stream);
    const auto cell_bytes                  = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);
    const GridHierarchy pressure_hierarchy = build_hierarchy(desc->nx, desc->ny, desc->nz, static_cast<float*>(desc->temporary_pressure), static_cast<float*>(desc->temporary_divergence), static_cast<float*>(desc->temporary_omega_x), static_cast<float*>(desc->temporary_omega_y));

    nvtx3::scoped_range range("vsmoke.step.project");
    if (cudaMemsetAsync(desc->temporary_pressure, 0, cell_bytes, stream) != cudaSuccess) return 5001;
    compute_poisson_rhs_kernel<<<cells, block, 0, stream>>>(static_cast<float*>(desc->temporary_divergence), static_cast<float*>(desc->temporary_previous_velocity_x), static_cast<float*>(desc->temporary_previous_velocity_y), static_cast<float*>(desc->temporary_previous_velocity_z), desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    const PoissonVCycleOps ops{};
    const VCycleConfig config{.cycles = std::max(1, desc->pressure_iterations / 40), .pre_smooth = 1, .post_smooth = 1, .coarse_smooth = std::max(8, desc->pressure_iterations / 10)};
    if (const int32_t code = run_v_cycle(pressure_hierarchy, config, ops, block, stream); code != 0) return code;
    project_velocity_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->temporary_previous_velocity_x), static_cast<float*>(desc->temporary_previous_velocity_y), static_cast<float*>(desc->temporary_previous_velocity_z), static_cast<float*>(desc->temporary_pressure), desc->nx, desc->ny, desc->nz, desc->dt / desc->cell_size);
    enforce_velocity_boundaries_kernel<<<velocity_grid, block, 0, stream>>>(static_cast<float*>(desc->temporary_previous_velocity_x), static_cast<float*>(desc->temporary_previous_velocity_y), static_cast<float*>(desc->temporary_previous_velocity_z), desc->nx, desc->ny, desc->nz, boundary_pack, desc->inflow_velocity_x_min,
        desc->inflow_velocity_x_max, desc->inflow_velocity_y_min, desc->inflow_velocity_y_max, desc->inflow_velocity_z_min, desc->inflow_velocity_z_max);
    if (cudaGetLastError() != cudaSuccess) return 5001;
    return 0;
}


int32_t visual_simulation_of_smoke_advect_scalar_flow_cuda(const VisualSimulationOfSmokeAdvectScalarFlowDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_advect_scalar_flow_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const dim3 cells      = make_grid(desc->nx, desc->ny, desc->nz, block);
    const bool cubic      = desc->use_monotonic_cubic != 0u;
    const uint32_t boundary_pack = make_boundary_pack(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max);
    const auto stream     = static_cast<Stream>(desc->stream);
    const auto cell_bytes = static_cast<std::uint64_t>(desc->nx) * static_cast<std::uint64_t>(desc->ny) * static_cast<std::uint64_t>(desc->nz) * sizeof(float);

    nvtx3::scoped_range range("vsmoke.step.advect_scalar_flow");
    for (int32_t scalar_index = 0; scalar_index < desc->scalar_count; ++scalar_index) {
        const auto& binding = desc->scalar_bindings[scalar_index];
        if (cudaMemcpyAsync(binding.temporary_previous_scalar, binding.scalar, cell_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return 5001;
        advect_scalar_kernel<<<cells, block, 0, stream>>>(
            static_cast<float*>(binding.scalar), static_cast<float*>(binding.temporary_previous_scalar), static_cast<float*>(desc->velocity_x), static_cast<float*>(desc->velocity_y), static_cast<float*>(desc->velocity_z), desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt, cubic, static_cast<int>(binding.clamp_non_negative));
        apply_scalar_inflow_boundaries_kernel<<<cells, block, 0, stream>>>(
            static_cast<float*>(binding.scalar), desc->nx, desc->ny, desc->nz, boundary_pack, binding.inflow_scalar_x_min, binding.inflow_scalar_x_max, binding.inflow_scalar_y_min, binding.inflow_scalar_y_max, binding.inflow_scalar_z_min, binding.inflow_scalar_z_max);
        if (cudaGetLastError() != cudaSuccess) return 5001;
    }
    return 0;
}

int32_t visual_simulation_of_smoke_add_scalar_source_cuda(const VisualSimulationOfSmokeAddScalarSourceDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_add_scalar_source_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const auto stream = static_cast<Stream>(desc->stream);
    add_scalar_source_kernel<<<make_grid(desc->nx, desc->ny, desc->nz, block), block, 0, stream>>>(static_cast<float*>(desc->scalar), desc->nx, desc->ny, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount, desc->sample_offset_x, desc->sample_offset_y, desc->sample_offset_z);
    return cudaGetLastError() == cudaSuccess ? 0 : 5001;
}

int32_t visual_simulation_of_smoke_add_vector_source_cuda(const VisualSimulationOfSmokeAddVectorSourceDesc* desc) {
    using namespace visual_smoke;
    if (const int32_t code = visual_simulation_of_smoke_validate_add_vector_source_desc(desc); code != 0) return code;

    const dim3 block(static_cast<unsigned>(std::max(desc->block_x, 1)), static_cast<unsigned>(std::max(desc->block_y, 1)), static_cast<unsigned>(std::max(desc->block_z, 1)));
    const auto stream = static_cast<Stream>(desc->stream);
    add_scalar_source_kernel<<<make_grid(desc->nx + 1, desc->ny, desc->nz, block), block, 0, stream>>>(static_cast<float*>(desc->vector_x), desc->nx + 1, desc->ny, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_x, 0.0f, 0.5f, 0.5f);
    add_scalar_source_kernel<<<make_grid(desc->nx, desc->ny + 1, desc->nz, block), block, 0, stream>>>(static_cast<float*>(desc->vector_y), desc->nx, desc->ny + 1, desc->nz, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_y, 0.5f, 0.0f, 0.5f);
    add_scalar_source_kernel<<<make_grid(desc->nx, desc->ny, desc->nz + 1, block), block, 0, stream>>>(static_cast<float*>(desc->vector_z), desc->nx, desc->ny, desc->nz + 1, desc->center_x, desc->center_y, desc->center_z, desc->radius, desc->amount_z, 0.5f, 0.5f, 0.0f);
    return cudaGetLastError() == cudaSuccess ? 0 : 5001;
}


} // extern "C"
