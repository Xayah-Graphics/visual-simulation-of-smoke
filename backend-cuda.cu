#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#include <nvtx3/nvtx3.hpp>

namespace visual_smoke {
    using Stream = cudaStream_t;

    namespace {

        [[nodiscard]] inline int32_t cuda_code(cudaError_t status) noexcept {
            return status == cudaSuccess ? 0 : 5001;
        }

        inline std::uint64_t scalar_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        inline std::uint64_t velocity_x_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        inline std::uint64_t velocity_y_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
        }

        inline std::uint64_t velocity_z_bytes(const int32_t nx, const int32_t ny, const int32_t nz) {
            return static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);
        }

        inline dim3 make_grid(int nx, int ny, int nz, const dim3& block) {
            return dim3(static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)), static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)), static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
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
                const std::uint64_t below = index_3d(i, j - 1, k, nx, ny);
                const std::uint64_t above = index_3d(i, j, k, nx, ny);
                const float density_avg = 0.5f * (density[below] + density[above]);
                const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
                const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
                const float buoyancy = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
                v[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
            }
            if (i < nx && j < ny && k > 0 && k < nz) w[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
        }

        __global__ void advect_velocity_kernel(float* dst_u, float* dst_v, float* dst_w, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz) {
                if (i == 0 || i == nx) dst_u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
                else {
                    const float3 pos = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                    const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_u[index_3d(i, j, k, nx + 1, ny)] = sample_u(src_u, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
            if (i < nx && j <= ny && k < nz) {
                if (j == 0 || j == ny) dst_v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
                else {
                    const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
                    const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_v[index_3d(i, j, k, nx, ny + 1)] = sample_v(src_v, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
            if (i < nx && j < ny && k <= nz) {
                if (k == 0 || k == nz) dst_w[index_3d(i, j, k, nx, ny)] = 0.0f;
                else {
                    const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
                    const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                    dst_w[index_3d(i, j, k, nx, ny)] = sample_w(src_w, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                }
            }
        }

        __global__ void compute_poisson_rhs_kernel(float* rhs, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) return;
            rhs[index_3d(i, j, k, nx, ny)] = -(fetch_clamped(u, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(u, i, j, k, nx + 1, ny, nz) + fetch_clamped(v, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(v, i, j, k, nx, ny + 1, nz)
                + fetch_clamped(w, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(w, i, j, k, nx, ny, nz + 1)) * (h / dt);
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
            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            const int coarse_nx = std::max(1, (fine_nx + 1) / 2);
            const int coarse_ny = std::max(1, (fine_ny + 1) / 2);
            const int coarse_nz = std::max(1, (fine_nz + 1) / 2);
            if (x >= coarse_nx || y >= coarse_ny || z >= coarse_nz) return;

            float residual_sum = 0.0f;
            int samples = 0;
            for (int fz = 2 * z; fz < std::min(2 * z + 2, fine_nz); ++fz) {
                for (int fy = 2 * y; fy < std::min(2 * y + 2, fine_ny); ++fy) {
                    for (int fx = 2 * x; fx < std::min(2 * x + 2, fine_nx); ++fx) {
                        float neighbors = 0.0f;
                        int count = 0;
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
                if (i == 0 || i == nx) u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
                else u[index_3d(i, j, k, nx + 1, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) * scale;
            }
            if (i < nx && j <= ny && k < nz) {
                if (j == 0 || j == ny) v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
                else v[index_3d(i, j, k, nx, ny + 1)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) * scale;
            }
            if (i < nx && j < ny && k <= nz) {
                if (k == 0 || k == nz) w[index_3d(i, j, k, nx, ny)] = 0.0f;
                else w[index_3d(i, j, k, nx, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) * scale;
            }
        }

        __global__ void advect_scalars_kernel(float* density_dst, float* temperature_dst, const float* density_src, const float* temperature_src, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k < nz) {
                const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                const float3 vel = sample_velocity(u, v, w, pos, nx, ny, nz, h, cubic);
                const float3 back = clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h);
                density_dst[index_3d(i, j, k, nx, ny)] = fmaxf(0.0f, sample_scalar(density_src, back, nx, ny, nz, h, cubic));
                temperature_dst[index_3d(i, j, k, nx, ny)] = sample_scalar(temperature_src, back, nx, ny, nz, h, cubic);
            }
        }

    } // namespace

} // namespace visual_smoke

extern "C" {

int32_t visual_simulation_of_smoke_validate_desc(const VisualSimulationOfSmokeStepDesc* desc) {
    if (desc == nullptr) return 1000;
    if (desc->struct_size < sizeof(VisualSimulationOfSmokeStepDesc)) return 1000;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->cell_size <= 0.0f) return 1002;
    if (desc->dt <= 0.0f) return 1003;
    if (desc->pressure_iterations <= 0) return 1004;
    if (desc->density == nullptr) return 2001;
    if (desc->temperature == nullptr) return 2002;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_previous_density == nullptr) return 2007;
    if (desc->temporary_previous_temperature == nullptr) return 2008;
    if (desc->temporary_previous_velocity_x == nullptr) return 2009;
    if (desc->temporary_previous_velocity_y == nullptr) return 2010;
    if (desc->temporary_previous_velocity_z == nullptr) return 2011;
    if (desc->temporary_pressure == nullptr) return 2012;
    if (desc->temporary_divergence == nullptr) return 2013;
    if (desc->temporary_omega_x == nullptr) return 2014;
    if (desc->temporary_omega_y == nullptr) return 2015;
    if (desc->temporary_omega_z == nullptr) return 2016;
    if (desc->temporary_omega_magnitude == nullptr) return 2017;
    if (desc->temporary_force_x == nullptr) return 2018;
    if (desc->temporary_force_y == nullptr) return 2019;
    if (desc->temporary_force_z == nullptr) return 2020;
    return 0;
}

int32_t visual_simulation_of_smoke_step_cuda(const VisualSimulationOfSmokeStepDesc* desc) {
    using namespace visual_smoke;
    const int32_t nx = desc->nx;
    const int32_t ny = desc->ny;
    const int32_t nz = desc->nz;
    const float cell_size = desc->cell_size;
    const float dt = desc->dt;
    const float ambient_temperature = desc->ambient_temperature;
    const float density_buoyancy = desc->density_buoyancy;
    const float temperature_buoyancy = desc->temperature_buoyancy;
    const float vorticity_epsilon = desc->vorticity_epsilon;
    const int32_t pressure_iterations = desc->pressure_iterations;
    const int32_t block_x = desc->block_x;
    const int32_t block_y = desc->block_y;
    const int32_t block_z = desc->block_z;
    const uint32_t use_monotonic_cubic = desc->use_monotonic_cubic;
    const auto cell_bytes = visual_smoke::scalar_bytes(nx, ny, nz);
    const auto u_bytes    = visual_smoke::velocity_x_bytes(nx, ny, nz);
    const auto v_bytes    = visual_smoke::velocity_y_bytes(nx, ny, nz);
    const auto w_bytes    = visual_smoke::velocity_z_bytes(nx, ny, nz);

    auto* density_prev     = static_cast<float*>(desc->temporary_previous_density);
    auto* temperature_prev = static_cast<float*>(desc->temporary_previous_temperature);
    auto* u_prev           = static_cast<float*>(desc->temporary_previous_velocity_x);
    auto* v_prev           = static_cast<float*>(desc->temporary_previous_velocity_y);
    auto* w_prev           = static_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure         = static_cast<float*>(desc->temporary_pressure);
    auto* divergence       = static_cast<float*>(desc->temporary_divergence);
    auto* omega_x          = static_cast<float*>(desc->temporary_omega_x);
    auto* omega_y          = static_cast<float*>(desc->temporary_omega_y);
    auto* omega_z          = static_cast<float*>(desc->temporary_omega_z);
    auto* omega_mag        = static_cast<float*>(desc->temporary_omega_magnitude);
    auto* force_x          = static_cast<float*>(desc->temporary_force_x);
    auto* force_y          = static_cast<float*>(desc->temporary_force_y);
    auto* force_z          = static_cast<float*>(desc->temporary_force_z);
    auto* density_f        = static_cast<float*>(desc->density);
    auto* temperature_f    = static_cast<float*>(desc->temperature);
    auto* u                = static_cast<float*>(desc->velocity_x);
    auto* v                = static_cast<float*>(desc->velocity_y);
    auto* w                = static_cast<float*>(desc->velocity_z);
    auto* coarse_pressure_storage = omega_x;
    auto* coarse_rhs_storage = omega_y;
    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    const dim3 cells = make_grid(nx, ny, nz, block);
    const dim3 velocity_grid = make_grid(nx + 1, ny + 1, nz + 1, block);
    const bool cubic = use_monotonic_cubic != 0u;
    const auto stream = reinterpret_cast<visual_smoke::Stream>(desc->stream);
    constexpr int max_levels = 16;
    int level_count = 1;
    int level_nx[max_levels]{nx};
    int level_ny[max_levels]{ny};
    int level_nz[max_levels]{nz};
    float* pressure_levels[max_levels]{pressure};
    float* rhs_levels[max_levels]{divergence};
    std::uint64_t coarse_offset = 0;
    while (level_count < max_levels && (level_nx[level_count - 1] > 1 || level_ny[level_count - 1] > 1 || level_nz[level_count - 1] > 1)) {
        level_nx[level_count] = std::max(1, (level_nx[level_count - 1] + 1) / 2);
        level_ny[level_count] = std::max(1, (level_ny[level_count - 1] + 1) / 2);
        level_nz[level_count] = std::max(1, (level_nz[level_count - 1] + 1) / 2);
        pressure_levels[level_count] = coarse_pressure_storage + coarse_offset;
        rhs_levels[level_count] = coarse_rhs_storage + coarse_offset;
        coarse_offset += static_cast<std::uint64_t>(level_nx[level_count]) * static_cast<std::uint64_t>(level_ny[level_count]) * static_cast<std::uint64_t>(level_nz[level_count]);
        ++level_count;
    }

    nvtx3::scoped_range step_range{"vsmoke.step"};
    {
        nvtx3::scoped_range range{"vsmoke.step.forces"};
        compute_vorticity_kernel<<<cells, block, 0, stream>>>(u, v, w, omega_x, omega_y, omega_z, omega_mag, nx, ny, nz, cell_size);
        compute_confinement_kernel<<<cells, block, 0, stream>>>(omega_x, omega_y, omega_z, omega_mag, force_x, force_y, force_z, nx, ny, nz, vorticity_epsilon, cell_size);
        apply_forces_kernel<<<velocity_grid, block, 0, stream>>>(u, v, w, density_f, temperature_f, force_x, force_y, force_z, nx, ny, nz, ambient_temperature, density_buoyancy, temperature_buoyancy, dt);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"vsmoke.step.advect_velocity"};
        advect_velocity_kernel<<<velocity_grid, block, 0, stream>>>(u_prev, v_prev, w_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"vsmoke.step.project"};
        if (cuda_code(cudaMemsetAsync(pressure, 0, cell_bytes, stream)) != 0) return 5001;
        compute_poisson_rhs_kernel<<<cells, block, 0, stream>>>(divergence, u_prev, v_prev, w_prev, nx, ny, nz, cell_size, dt);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
        const int v_cycles = std::max(1, pressure_iterations / 40);
        const int smoothing_steps = 1;
        const int coarse_steps = std::max(8, pressure_iterations / 10);
        for (int cycle = 0; cycle < v_cycles; ++cycle) {
            for (int level = 0; level + 1 < level_count; ++level) {
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const dim3 level_grid = make_grid(lx, ly, lz, block);
                for (int smooth = 0; smooth < smoothing_steps; ++smooth) {
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0);
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1);
                }
                const int cx = level_nx[level + 1];
                const int cy = level_ny[level + 1];
                const int cz = level_nz[level + 1];
                const auto coarse_bytes = static_cast<std::uint64_t>(cx) * static_cast<std::uint64_t>(cy) * static_cast<std::uint64_t>(cz) * sizeof(float);
                if (cuda_code(cudaMemsetAsync(pressure_levels[level + 1], 0, coarse_bytes, stream)) != 0) return 5001;
                restrict_poisson_residual_kernel<<<make_grid(cx, cy, cz, block), block, 0, stream>>>(rhs_levels[level + 1], pressure_levels[level], rhs_levels[level], lx, ly, lz);
            }
            {
                const int level = level_count - 1;
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const dim3 level_grid = make_grid(lx, ly, lz, block);
                for (int smooth = 0; smooth < coarse_steps; ++smooth) {
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0);
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1);
                }
            }
            for (int level = level_count - 2; level >= 0; --level) {
                const int lx = level_nx[level];
                const int ly = level_ny[level];
                const int lz = level_nz[level];
                const int cx = level_nx[level + 1];
                const int cy = level_ny[level + 1];
                const int cz = level_nz[level + 1];
                const dim3 level_grid = make_grid(lx, ly, lz, block);
                prolongate_add_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], pressure_levels[level + 1], lx, ly, lz, cx, cy, cz);
                for (int smooth = 0; smooth < smoothing_steps; ++smooth) {
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 0);
                    poisson_rbgs_kernel<<<level_grid, block, 0, stream>>>(pressure_levels[level], rhs_levels[level], lx, ly, lz, 1);
                }
            }
        }
        project_velocity_kernel<<<velocity_grid, block, 0, stream>>>(u_prev, v_prev, w_prev, pressure, nx, ny, nz, dt / cell_size);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    if (cuda_code(cudaMemcpyAsync(u, u_prev, u_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(v, v_prev, v_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(w, w_prev, w_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(density_prev, density_f, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(temperature_prev, temperature_f, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    {
        nvtx3::scoped_range range{"vsmoke.step.advect_scalars"};
        advect_scalars_kernel<<<cells, block, 0, stream>>>(density_f, temperature_f, density_prev, temperature_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    return 0;
}

} // extern "C"
