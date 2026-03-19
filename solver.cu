#include "visual-simulation-of-smoke.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace visual_smoke {
    using Desc = VisualSimulationOfSmokeContextDesc;
    using BufferView = VisualSimulationOfSmokeBufferView;
    using Source = VisualSimulationOfSmokeSourceDesc;
    using Stream = cudaStream_t;

    thread_local std::string g_last_error{};

    namespace {

        inline void check_cuda(cudaError_t status, const char* what) {
            if (status == cudaSuccess) {
                return;
            }
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }

        inline dim3 make_block(const Desc& desc) {
            return dim3(
                static_cast<unsigned>(std::max(desc.block_x, 1)),
                static_cast<unsigned>(std::max(desc.block_y, 1)),
                static_cast<unsigned>(std::max(desc.block_z, 1)));
        }

        inline dim3 make_grid(int nx, int ny, int nz, const dim3& block) {
            return dim3(
                static_cast<unsigned>((nx + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
                static_cast<unsigned>((ny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)),
                static_cast<unsigned>((nz + static_cast<int>(block.z) - 1) / static_cast<int>(block.z)));
        }

        __host__ __device__ inline int clampi(int value, int lo, int hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ inline float clampf(float value, float lo, float hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ inline std::uint64_t index_3d(int x, int y, int z, int sx, int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) +
                   static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) +
                   static_cast<std::uint64_t>(x);
        }

        __device__ inline float fetch_clamped(const float* field, int x, int y, int z, int sx, int sy, int sz) {
            return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
        }

        __device__ inline float monotonic_cubic(float p0, float p1, float p2, float p3, float t) {
            const float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
            const float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
            const float a2 = -0.5f * p0 + 0.5f * p2;
            const float a3 = p1;
            return clampf(((a0 * t + a1) * t + a2) * t + a3, fminf(p1, p2), fmaxf(p1, p2));
        }

        __device__ float sample_grid(const float* field, float gx, float gy, float gz, int sx, int sy, int sz, bool cubic) {
            gx = clampf(gx, 0.0f, static_cast<float>(sx - 1));
            gy = clampf(gy, 0.0f, static_cast<float>(sy - 1));
            gz = clampf(gz, 0.0f, static_cast<float>(sz - 1));

            if (!cubic) {
                const int x0 = clampi(static_cast<int>(floorf(gx)), 0, sx - 1);
                const int y0 = clampi(static_cast<int>(floorf(gy)), 0, sy - 1);
                const int z0 = clampi(static_cast<int>(floorf(gz)), 0, sz - 1);
                const int x1 = min(x0 + 1, sx - 1);
                const int y1 = min(y0 + 1, sy - 1);
                const int z1 = min(z0 + 1, sz - 1);
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
                const float c0 = c00 + (c10 - c00) * ty;
                const float c1 = c01 + (c11 - c01) * ty;
                return c0 + (c1 - c0) * tz;
            }

            const int ix = static_cast<int>(floorf(gx));
            const int iy = static_cast<int>(floorf(gy));
            const int iz = static_cast<int>(floorf(gz));
            const float tx = gx - static_cast<float>(ix);
            const float ty = gy - static_cast<float>(iy);
            const float tz = gz - static_cast<float>(iz);

            float yz[4][4];
            for (int zz = 0; zz < 4; ++zz) {
                for (int yy = 0; yy < 4; ++yy) {
                    yz[zz][yy] = monotonic_cubic(
                        fetch_clamped(field, ix - 1, iy + yy - 1, iz + zz - 1, sx, sy, sz),
                        fetch_clamped(field, ix + 0, iy + yy - 1, iz + zz - 1, sx, sy, sz),
                        fetch_clamped(field, ix + 1, iy + yy - 1, iz + zz - 1, sx, sy, sz),
                        fetch_clamped(field, ix + 2, iy + yy - 1, iz + zz - 1, sx, sy, sz),
                        tx);
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
            return make_float3(
                clampf(pos.x, 0.0f, static_cast<float>(nx) * h),
                clampf(pos.y, 0.0f, static_cast<float>(ny) * h),
                clampf(pos.z, 0.0f, static_cast<float>(nz) * h));
        }

        __device__ float3 sample_velocity(const float* u, const float* v, const float* w, float3 pos, int nx, int ny, int nz, float h, bool cubic) {
            pos = clamp_domain(pos, nx, ny, nz, h);
            return make_float3(sample_u(u, pos, nx, ny, nz, h, cubic), sample_v(v, pos, nx, ny, nz, h, cubic), sample_w(w, pos, nx, ny, nz, h, cubic));
        }

        __device__ float center_u(const float* u, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = clampi(i, 0, nx - 1);
            const int cj = clampi(j, 0, ny - 1);
            const int ck = clampi(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(u, ci, cj, ck, nx + 1, ny, nz) + fetch_clamped(u, ci + 1, cj, ck, nx + 1, ny, nz));
        }

        __device__ float center_v(const float* v, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = clampi(i, 0, nx - 1);
            const int cj = clampi(j, 0, ny - 1);
            const int ck = clampi(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(v, ci, cj, ck, nx, ny + 1, nz) + fetch_clamped(v, ci, cj + 1, ck, nx, ny + 1, nz));
        }

        __device__ float center_w(const float* w, int i, int j, int k, int nx, int ny, int nz) {
            const int ci = clampi(i, 0, nx - 1);
            const int cj = clampi(j, 0, ny - 1);
            const int ck = clampi(k, 0, nz - 1);
            return 0.5f * (fetch_clamped(w, ci, cj, ck, nx, ny, nz + 1) + fetch_clamped(w, ci, cj, ck + 1, nx, ny, nz + 1));
        }

        __global__ void add_source_cells_kernel(float* density, float* temperature, int nx, int ny, int nz, float cx, float cy, float cz, float radius, float density_amount, float temperature_amount) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) {
                return;
            }

            const float dx = (static_cast<float>(i) + 0.5f) - cx;
            const float dy = (static_cast<float>(j) + 0.5f) - cy;
            const float dz = (static_cast<float>(k) + 0.5f) - cz;
            const float radius_sq = radius * radius;
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq > radius_sq) {
                return;
            }

            const float weight = fmaxf(0.0f, 1.0f - dist_sq / radius_sq);
            const std::uint64_t index = index_3d(i, j, k, nx, ny);
            density[index] += density_amount * weight;
            temperature[index] += temperature_amount * weight;
        }

        __global__ void add_source_u_kernel(float* u, int nx, int ny, int nz, float cx, float cy, float cz, float radius, float amount) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > nx || j >= ny || k >= nz) {
                return;
            }
            const float dx = static_cast<float>(i) - cx;
            const float dy = (static_cast<float>(j) + 0.5f) - cy;
            const float dz = (static_cast<float>(k) + 0.5f) - cz;
            const float radius_sq = radius * radius;
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq <= radius_sq) {
                u[index_3d(i, j, k, nx + 1, ny)] += amount * fmaxf(0.0f, 1.0f - dist_sq / radius_sq);
            }
        }

        __global__ void add_source_v_kernel(float* v, int nx, int ny, int nz, float cx, float cy, float cz, float radius, float amount) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j > ny || k >= nz) {
                return;
            }
            const float dx = (static_cast<float>(i) + 0.5f) - cx;
            const float dy = static_cast<float>(j) - cy;
            const float dz = (static_cast<float>(k) + 0.5f) - cz;
            const float radius_sq = radius * radius;
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq <= radius_sq) {
                v[index_3d(i, j, k, nx, ny + 1)] += amount * fmaxf(0.0f, 1.0f - dist_sq / radius_sq);
            }
        }

        __global__ void add_source_w_kernel(float* w, int nx, int ny, int nz, float cx, float cy, float cz, float radius, float amount) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k > nz) {
                return;
            }
            const float dx = (static_cast<float>(i) + 0.5f) - cx;
            const float dy = (static_cast<float>(j) + 0.5f) - cy;
            const float dz = static_cast<float>(k) - cz;
            const float radius_sq = radius * radius;
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq <= radius_sq) {
                w[index_3d(i, j, k, nx, ny)] += amount * fmaxf(0.0f, 1.0f - dist_sq / radius_sq);
            }
        }

        __global__ void set_u_boundary_kernel(float* u, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz && (i == 0 || i == nx)) {
                u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
            }
        }

        __global__ void set_v_boundary_kernel(float* v, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j <= ny && k < nz && (j == 0 || j == ny)) {
                v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
            }
        }

        __global__ void set_w_boundary_kernel(float* w, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k <= nz && (k == 0 || k == nz)) {
                w[index_3d(i, j, k, nx, ny)] = 0.0f;
            }
        }

        __global__ void compute_vorticity_kernel(const float* u, const float* v, const float* w, float* omega_x, float* omega_y, float* omega_z, float* omega_mag, int nx, int ny, int nz, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) {
                return;
            }

            const float dw_dy = (center_w(w, i, j + 1, k, nx, ny, nz) - center_w(w, i, j - 1, k, nx, ny, nz)) / (2.0f * h);
            const float dv_dz = (center_v(v, i, j, k + 1, nx, ny, nz) - center_v(v, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
            const float du_dz = (center_u(u, i, j, k + 1, nx, ny, nz) - center_u(u, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
            const float dw_dx = (center_w(w, i + 1, j, k, nx, ny, nz) - center_w(w, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
            const float dv_dx = (center_v(v, i + 1, j, k, nx, ny, nz) - center_v(v, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
            const float du_dy = (center_u(u, i, j + 1, k, nx, ny, nz) - center_u(u, i, j - 1, k, nx, ny, nz)) / (2.0f * h);

            const float wx = dw_dy - dv_dz;
            const float wy = du_dz - dw_dx;
            const float wz = dv_dx - du_dy;
            const std::uint64_t index = index_3d(i, j, k, nx, ny);
            omega_x[index] = wx;
            omega_y[index] = wy;
            omega_z[index] = wz;
            omega_mag[index] = sqrtf(wx * wx + wy * wy + wz * wz);
        }

        __global__ void compute_confinement_kernel(const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_mag, float* force_x, float* force_y, float* force_z, int nx, int ny, int nz, float epsilon, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) {
                return;
            }

            const auto mag = [&](int x, int y, int z) {
                return fetch_clamped(omega_mag, x, y, z, nx, ny, nz);
            };

            const float gx = (mag(i + 1, j, k) - mag(i - 1, j, k)) / (2.0f * h);
            const float gy = (mag(i, j + 1, k) - mag(i, j - 1, k)) / (2.0f * h);
            const float gz = (mag(i, j, k + 1) - mag(i, j, k - 1)) / (2.0f * h);
            const float inv_len = rsqrtf(fmaxf(gx * gx + gy * gy + gz * gz, 1.0e-12f));
            const float nxv = gx * inv_len;
            const float nyv = gy * inv_len;
            const float nzv = gz * inv_len;
            const std::uint64_t index = index_3d(i, j, k, nx, ny);
            force_x[index] = epsilon * h * (nyv * omega_z[index] - nzv * omega_y[index]);
            force_y[index] = epsilon * h * (nzv * omega_x[index] - nxv * omega_z[index]);
            force_z[index] = epsilon * h * (nxv * omega_y[index] - nyv * omega_x[index]);
        }

        __global__ void apply_u_forces_kernel(float* u, const float* force_x, int nx, int ny, int nz, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > 0 && i < nx && j < ny && k < nz) {
                u[index_3d(i, j, k, nx + 1, ny)] += 0.5f * dt * (force_x[index_3d(i - 1, j, k, nx, ny)] + force_x[index_3d(i, j, k, nx, ny)]);
            }
        }

        __global__ void apply_v_forces_kernel(float* v, const float* density, const float* temperature, const float* force_y, int nx, int ny, int nz, float ambient_temperature, float density_buoyancy, float temperature_buoyancy, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j <= 0 || j >= ny || k >= nz) {
                return;
            }

            const std::uint64_t below = index_3d(i, j - 1, k, nx, ny);
            const std::uint64_t above = index_3d(i, j, k, nx, ny);
            const float density_avg = 0.5f * (density[below] + density[above]);
            const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
            const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
            const float buoyancy = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
            v[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
        }

        __global__ void apply_w_forces_kernel(float* w, const float* force_z, int nx, int ny, int nz, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k <= 0 || k >= nz) {
                return;
            }
            w[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
        }

        __global__ void advect_u_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz) {
                const float3 pos = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx + 1, ny)] = sample_u(src_u, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
            }
        }

        __global__ void advect_v_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j <= ny && k < nz) {
                const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
                const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx, ny + 1)] = sample_v(src_v, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
            }
        }

        __global__ void advect_w_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k <= nz) {
                const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
                const float3 vel = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx, ny)] = sample_w(src_w, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
            }
        }

        __global__ void compute_divergence_kernel(float* divergence, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k < nz) {
                divergence[index_3d(i, j, k, nx, ny)] =
                    (fetch_clamped(u, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(u, i, j, k, nx + 1, ny, nz) +
                     fetch_clamped(v, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(v, i, j, k, nx, ny + 1, nz) +
                     fetch_clamped(w, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(w, i, j, k, nx, ny, nz + 1)) / h;
            }
        }

        __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, int nx, int ny, int nz, float h, float dt, int parity) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz || ((i + j + k) & 1) != parity) {
                return;
            }

            float sum = 0.0f;
            int count = 0;
            if (i > 0) { sum += pressure[index_3d(i - 1, j, k, nx, ny)]; ++count; }
            if (i + 1 < nx) { sum += pressure[index_3d(i + 1, j, k, nx, ny)]; ++count; }
            if (j > 0) { sum += pressure[index_3d(i, j - 1, k, nx, ny)]; ++count; }
            if (j + 1 < ny) { sum += pressure[index_3d(i, j + 1, k, nx, ny)]; ++count; }
            if (k > 0) { sum += pressure[index_3d(i, j, k - 1, nx, ny)]; ++count; }
            if (k + 1 < nz) { sum += pressure[index_3d(i, j, k + 1, nx, ny)]; ++count; }

            pressure[index_3d(i, j, k, nx, ny)] = count > 0 ? (sum - divergence[index_3d(i, j, k, nx, ny)] * h * h / dt) / static_cast<float>(count) : 0.0f;
        }

        __global__ void subtract_gradient_u_kernel(float* u, const float* pressure, int nx, int ny, int nz, float h, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > 0 && i < nx && j < ny && k < nz) {
                u[index_3d(i, j, k, nx + 1, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) / h;
            }
        }

        __global__ void subtract_gradient_v_kernel(float* v, const float* pressure, int nx, int ny, int nz, float h, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j > 0 && j < ny && k < nz) {
                v[index_3d(i, j, k, nx, ny + 1)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) / h;
            }
        }

        __global__ void subtract_gradient_w_kernel(float* w, const float* pressure, int nx, int ny, int nz, float h, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k > 0 && k < nz) {
                w[index_3d(i, j, k, nx, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) / h;
            }
        }

        __global__ void advect_scalar_kernel(float* dst, const float* src, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h, float dt, bool cubic, bool clamp_nonnegative) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k < nz) {
                const float3 pos = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                const float3 vel = sample_velocity(u, v, w, pos, nx, ny, nz, h, cubic);
                float value = sample_scalar(src, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx, ny)] = clamp_nonnegative ? fmaxf(0.0f, value) : value;
            }
        }

        __global__ void snapshot_velocity_magnitude_kernel(float* dst, const float* u, const float* v, const float* w, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k >= nz) {
                return;
            }

            const float ux = center_u(u, i, j, k, nx, ny, nz);
            const float vy = center_v(v, i, j, k, nx, ny, nz);
            const float wz = center_w(w, i, j, k, nx, ny, nz);
            dst[index_3d(i, j, k, nx, ny)] = sqrtf(ux * ux + vy * vy + wz * wz);
        }

    } // namespace

    class Solver {
    public:
        explicit Solver(const Desc& desc);
        ~Solver();
        Solver(const Solver&) = delete;
        Solver& operator=(const Solver&) = delete;

        [[nodiscard]] std::uint64_t density_bytes() const noexcept { return cell_bytes_; }
        [[nodiscard]] std::uint64_t temperature_bytes() const noexcept { return cell_bytes_; }

        void clear(Stream stream);
        void add_source(const Source& source, Stream stream);
        void step(Stream stream);
        void snapshot_density(const BufferView& destination, Stream stream);
        void snapshot_temperature(const BufferView& destination, Stream stream);
        void snapshot_velocity_magnitude(const BufferView& destination, Stream stream);

    private:
        void validate_snapshot_(const BufferView& destination, const char* name) const;

        Desc desc_{};
        std::uint64_t cell_count_ = 0;
        std::uint64_t u_count_ = 0;
        std::uint64_t v_count_ = 0;
        std::uint64_t w_count_ = 0;
        std::uint64_t cell_bytes_ = 0;
        float* density_ = nullptr;
        float* density_prev_ = nullptr;
        float* temperature_ = nullptr;
        float* temperature_prev_ = nullptr;
        float* u_ = nullptr;
        float* u_prev_ = nullptr;
        float* v_ = nullptr;
        float* v_prev_ = nullptr;
        float* w_ = nullptr;
        float* w_prev_ = nullptr;
        float* pressure_ = nullptr;
        float* divergence_ = nullptr;
        float* omega_x_ = nullptr;
        float* omega_y_ = nullptr;
        float* omega_z_ = nullptr;
        float* omega_mag_ = nullptr;
        float* force_x_ = nullptr;
        float* force_y_ = nullptr;
        float* force_z_ = nullptr;
    };

    Solver::Solver(const Desc& desc) : desc_(desc) {
        if (desc_.nx <= 0 || desc_.ny <= 0 || desc_.nz <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
        if (desc_.dt <= 0.0f || desc_.cell_size <= 0.0f) {
            throw std::invalid_argument("dt and cell_size must be positive");
        }
        if (desc_.pressure_iterations <= 0) {
            throw std::invalid_argument("pressure_iterations must be positive");
        }

        cell_count_ = static_cast<std::uint64_t>(desc_.nx) * static_cast<std::uint64_t>(desc_.ny) * static_cast<std::uint64_t>(desc_.nz);
        u_count_ = static_cast<std::uint64_t>(desc_.nx + 1) * static_cast<std::uint64_t>(desc_.ny) * static_cast<std::uint64_t>(desc_.nz);
        v_count_ = static_cast<std::uint64_t>(desc_.nx) * static_cast<std::uint64_t>(desc_.ny + 1) * static_cast<std::uint64_t>(desc_.nz);
        w_count_ = static_cast<std::uint64_t>(desc_.nx) * static_cast<std::uint64_t>(desc_.ny) * static_cast<std::uint64_t>(desc_.nz + 1);
        cell_bytes_ = cell_count_ * sizeof(float);

        auto allocate = [](float*& field, std::uint64_t count) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&field), count * sizeof(float)), "cudaMalloc");
        };

        allocate(density_, cell_count_);
        allocate(density_prev_, cell_count_);
        allocate(temperature_, cell_count_);
        allocate(temperature_prev_, cell_count_);
        allocate(u_, u_count_);
        allocate(u_prev_, u_count_);
        allocate(v_, v_count_);
        allocate(v_prev_, v_count_);
        allocate(w_, w_count_);
        allocate(w_prev_, w_count_);
        allocate(pressure_, cell_count_);
        allocate(divergence_, cell_count_);
        allocate(omega_x_, cell_count_);
        allocate(omega_y_, cell_count_);
        allocate(omega_z_, cell_count_);
        allocate(omega_mag_, cell_count_);
        allocate(force_x_, cell_count_);
        allocate(force_y_, cell_count_);
        allocate(force_z_, cell_count_);

        clear(nullptr);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize constructor");
    }

    Solver::~Solver() {
        cudaFree(density_);
        cudaFree(density_prev_);
        cudaFree(temperature_);
        cudaFree(temperature_prev_);
        cudaFree(u_);
        cudaFree(u_prev_);
        cudaFree(v_);
        cudaFree(v_prev_);
        cudaFree(w_);
        cudaFree(w_prev_);
        cudaFree(pressure_);
        cudaFree(divergence_);
        cudaFree(omega_x_);
        cudaFree(omega_y_);
        cudaFree(omega_z_);
        cudaFree(omega_mag_);
        cudaFree(force_x_);
        cudaFree(force_y_);
        cudaFree(force_z_);
    }

    void Solver::validate_snapshot_(const BufferView& destination, const char* name) const {
        if (destination.data == nullptr) {
            throw std::invalid_argument(std::string(name) + " must provide a non-null device pointer");
        }
        if (destination.format != VISUAL_SIMULATION_OF_SMOKE_BUFFER_FORMAT_F32) {
            throw std::invalid_argument(std::string(name) + " must use VISUAL_SIMULATION_OF_SMOKE_BUFFER_FORMAT_F32");
        }
        if (destination.memory_type != VISUAL_SIMULATION_OF_SMOKE_MEMORY_TYPE_CUDA_DEVICE) {
            throw std::invalid_argument(std::string(name) + " must use VISUAL_SIMULATION_OF_SMOKE_MEMORY_TYPE_CUDA_DEVICE");
        }
        if (destination.size_bytes < cell_bytes_) {
            throw std::invalid_argument(std::string(name) + " size_bytes is too small");
        }
    }

    void Solver::clear(Stream stream) {
        nvtx3::scoped_range range{"vsmoke.clear"};
        check_cuda(cudaMemsetAsync(density_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync density");
        check_cuda(cudaMemsetAsync(density_prev_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync density_prev");
        check_cuda(cudaMemsetAsync(temperature_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync temperature");
        check_cuda(cudaMemsetAsync(temperature_prev_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync temperature_prev");
        check_cuda(cudaMemsetAsync(u_, 0, u_count_ * sizeof(float), stream), "cudaMemsetAsync u");
        check_cuda(cudaMemsetAsync(u_prev_, 0, u_count_ * sizeof(float), stream), "cudaMemsetAsync u_prev");
        check_cuda(cudaMemsetAsync(v_, 0, v_count_ * sizeof(float), stream), "cudaMemsetAsync v");
        check_cuda(cudaMemsetAsync(v_prev_, 0, v_count_ * sizeof(float), stream), "cudaMemsetAsync v_prev");
        check_cuda(cudaMemsetAsync(w_, 0, w_count_ * sizeof(float), stream), "cudaMemsetAsync w");
        check_cuda(cudaMemsetAsync(w_prev_, 0, w_count_ * sizeof(float), stream), "cudaMemsetAsync w_prev");
        check_cuda(cudaMemsetAsync(pressure_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync pressure");
        check_cuda(cudaMemsetAsync(divergence_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync divergence");
        check_cuda(cudaMemsetAsync(omega_x_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync omega_x");
        check_cuda(cudaMemsetAsync(omega_y_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync omega_y");
        check_cuda(cudaMemsetAsync(omega_z_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync omega_z");
        check_cuda(cudaMemsetAsync(omega_mag_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync omega_mag");
        check_cuda(cudaMemsetAsync(force_x_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync force_x");
        check_cuda(cudaMemsetAsync(force_y_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync force_y");
        check_cuda(cudaMemsetAsync(force_z_, 0, cell_count_ * sizeof(float), stream), "cudaMemsetAsync force_z");
    }

    void Solver::add_source(const Source& source, Stream stream) {
        nvtx3::scoped_range range{"vsmoke.add_source"};
        if (source.radius <= 0.0f) {
            throw std::invalid_argument("source radius must be positive");
        }

        const dim3 block = make_block(desc_);
        add_source_cells_kernel<<<make_grid(desc_.nx, desc_.ny, desc_.nz, block), block, 0, stream>>>(
            density_, temperature_, desc_.nx, desc_.ny, desc_.nz,
            source.center_x, source.center_y, source.center_z, source.radius, source.density_amount, source.temperature_amount);
        add_source_u_kernel<<<make_grid(desc_.nx + 1, desc_.ny, desc_.nz, block), block, 0, stream>>>(
            u_, desc_.nx, desc_.ny, desc_.nz, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_x);
        add_source_v_kernel<<<make_grid(desc_.nx, desc_.ny + 1, desc_.nz, block), block, 0, stream>>>(
            v_, desc_.nx, desc_.ny, desc_.nz, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_y);
        add_source_w_kernel<<<make_grid(desc_.nx, desc_.ny, desc_.nz + 1, block), block, 0, stream>>>(
            w_, desc_.nx, desc_.ny, desc_.nz, source.center_x, source.center_y, source.center_z, source.radius, source.velocity_z);
        check_cuda(cudaGetLastError(), "add_source kernels");
    }

    void Solver::step(Stream stream) {
        nvtx3::scoped_range step_range{"vsmoke.step"};
        const dim3 block = make_block(desc_);
        const bool cubic = desc_.use_monotonic_cubic != 0u;
        const dim3 cells = make_grid(desc_.nx, desc_.ny, desc_.nz, block);
        const dim3 u_grid = make_grid(desc_.nx + 1, desc_.ny, desc_.nz, block);
        const dim3 v_grid = make_grid(desc_.nx, desc_.ny + 1, desc_.nz, block);
        const dim3 w_grid = make_grid(desc_.nx, desc_.ny, desc_.nz + 1, block);

        {
            nvtx3::scoped_range range{"vsmoke.step.forces"};
            compute_vorticity_kernel<<<cells, block, 0, stream>>>(u_, v_, w_, omega_x_, omega_y_, omega_z_, omega_mag_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size);
            compute_confinement_kernel<<<cells, block, 0, stream>>>(omega_x_, omega_y_, omega_z_, omega_mag_, force_x_, force_y_, force_z_, desc_.nx, desc_.ny, desc_.nz, desc_.vorticity_epsilon, desc_.cell_size);
            apply_u_forces_kernel<<<u_grid, block, 0, stream>>>(u_, force_x_, desc_.nx, desc_.ny, desc_.nz, desc_.dt);
            apply_v_forces_kernel<<<v_grid, block, 0, stream>>>(v_, density_, temperature_, force_y_, desc_.nx, desc_.ny, desc_.nz, desc_.ambient_temperature, desc_.density_buoyancy, desc_.temperature_buoyancy, desc_.dt);
            apply_w_forces_kernel<<<w_grid, block, 0, stream>>>(w_, force_z_, desc_.nx, desc_.ny, desc_.nz, desc_.dt);
        }

        std::swap(u_, u_prev_);
        std::swap(v_, v_prev_);
        std::swap(w_, w_prev_);

        {
            nvtx3::scoped_range range{"vsmoke.step.advect_velocity"};
            advect_u_kernel<<<u_grid, block, 0, stream>>>(u_, u_prev_, v_prev_, w_prev_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, cubic);
            advect_v_kernel<<<v_grid, block, 0, stream>>>(v_, u_prev_, v_prev_, w_prev_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, cubic);
            advect_w_kernel<<<w_grid, block, 0, stream>>>(w_, u_prev_, v_prev_, w_prev_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, cubic);
            set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(u_, desc_.nx, desc_.ny, desc_.nz);
            set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(v_, desc_.nx, desc_.ny, desc_.nz);
            set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(w_, desc_.nx, desc_.ny, desc_.nz);
        }

        {
            nvtx3::scoped_range range{"vsmoke.step.project"};
            check_cuda(cudaMemsetAsync(pressure_, 0, cell_bytes_, stream), "cudaMemsetAsync pressure");
            compute_divergence_kernel<<<cells, block, 0, stream>>>(divergence_, u_, v_, w_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size);
            for (int iteration = 0; iteration < desc_.pressure_iterations; ++iteration) {
                pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure_, divergence_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, 0);
                pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure_, divergence_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, 1);
            }

            subtract_gradient_u_kernel<<<u_grid, block, 0, stream>>>(u_, pressure_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt);
            subtract_gradient_v_kernel<<<v_grid, block, 0, stream>>>(v_, pressure_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt);
            subtract_gradient_w_kernel<<<w_grid, block, 0, stream>>>(w_, pressure_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt);
            set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(u_, desc_.nx, desc_.ny, desc_.nz);
            set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(v_, desc_.nx, desc_.ny, desc_.nz);
            set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(w_, desc_.nx, desc_.ny, desc_.nz);
        }

        std::swap(density_, density_prev_);
        std::swap(temperature_, temperature_prev_);
        {
            nvtx3::scoped_range range{"vsmoke.step.advect_scalars"};
            advect_scalar_kernel<<<cells, block, 0, stream>>>(density_, density_prev_, u_, v_, w_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, cubic, true);
            advect_scalar_kernel<<<cells, block, 0, stream>>>(temperature_, temperature_prev_, u_, v_, w_, desc_.nx, desc_.ny, desc_.nz, desc_.cell_size, desc_.dt, cubic, false);
        }
        check_cuda(cudaGetLastError(), "step kernels");
    }

    void Solver::snapshot_density(const BufferView& destination, Stream stream) {
        nvtx3::scoped_range range{"vsmoke.snapshot_density"};
        validate_snapshot_(destination, "snapshot density destination");
        check_cuda(cudaMemcpyAsync(destination.data, density_, cell_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync density snapshot");
    }

    void Solver::snapshot_temperature(const BufferView& destination, Stream stream) {
        nvtx3::scoped_range range{"vsmoke.snapshot_temperature"};
        validate_snapshot_(destination, "snapshot temperature destination");
        check_cuda(cudaMemcpyAsync(destination.data, temperature_, cell_bytes_, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync temperature snapshot");
    }

    void Solver::snapshot_velocity_magnitude(const BufferView& destination, Stream stream) {
        nvtx3::scoped_range range{"vsmoke.snapshot_velocity_magnitude"};
        validate_snapshot_(destination, "snapshot velocity magnitude destination");
        const dim3 block = make_block(desc_);
        snapshot_velocity_magnitude_kernel<<<make_grid(desc_.nx, desc_.ny, desc_.nz, block), block, 0, stream>>>(reinterpret_cast<float*>(destination.data), u_, v_, w_, desc_.nx, desc_.ny, desc_.nz);
        check_cuda(cudaGetLastError(), "snapshot_velocity_magnitude_kernel");
    }

} // namespace visual_smoke

struct VisualSimulationOfSmokeContext {
    visual_smoke::Solver* solver = nullptr;
    std::string last_error{};
};

namespace {

    [[nodiscard]] visual_smoke::Stream to_stream(void* cuda_stream) noexcept {
        return reinterpret_cast<visual_smoke::Stream>(cuda_stream);
    }

    void set_global_error(const char* message) {
        visual_smoke::g_last_error = message != nullptr ? message : "unknown visual-simulation-of-smoke error";
    }

    int32_t store_error(VisualSimulationOfSmokeContext* context, int32_t code, const char* message) {
        set_global_error(message);
        if (context != nullptr) {
            context->last_error = visual_smoke::g_last_error;
        }
        return code;
    }

    int32_t copy_error_text(const std::string& text, char* buffer, std::uint64_t buffer_size) {
        if (buffer == nullptr || buffer_size == 0) {
            return VISUAL_SIMULATION_OF_SMOKE_ERROR_BUFFER_TOO_SMALL;
        }
        if (buffer_size <= static_cast<std::uint64_t>(text.size())) {
            buffer[0] = '\0';
            return VISUAL_SIMULATION_OF_SMOKE_ERROR_BUFFER_TOO_SMALL;
        }
        std::memcpy(buffer, text.c_str(), text.size() + 1);
        return VISUAL_SIMULATION_OF_SMOKE_SUCCESS;
    }

    template <class Fn>
    int32_t smoke_try(VisualSimulationOfSmokeContext* context, Fn&& fn) {
        try {
            fn();
            visual_smoke::g_last_error.clear();
            if (context != nullptr) {
                context->last_error.clear();
            }
            return VISUAL_SIMULATION_OF_SMOKE_SUCCESS;
        } catch (const std::bad_alloc& ex) {
            return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_ALLOCATION_FAILED, ex.what());
        } catch (const std::invalid_argument& ex) {
            return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, ex.what());
        } catch (const std::exception& ex) {
            return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME, ex.what());
        } catch (...) {
            return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME, "unknown visual-simulation-of-smoke exception");
        }
    }

} // namespace

extern "C" {

VisualSimulationOfSmokeContextDesc visual_simulation_of_smoke_context_desc_default(void) {
    return VisualSimulationOfSmokeContextDesc{
        .nx = 64,
        .ny = 96,
        .nz = 64,
        .dt = 1.0f / 90.0f,
        .cell_size = 1.0f,
        .ambient_temperature = 0.0f,
        .density_buoyancy = 0.045f,
        .temperature_buoyancy = 0.12f,
        .vorticity_epsilon = 2.0f,
        .pressure_iterations = 80,
        .block_x = 8,
        .block_y = 8,
        .block_z = 4,
        .use_monotonic_cubic = 1u,
    };
}

VisualSimulationOfSmokeContext* visual_simulation_of_smoke_context_create(const VisualSimulationOfSmokeContextDesc* desc) {
    if (desc == nullptr) {
        set_global_error("visual_simulation_of_smoke_context_create received a null desc");
        return nullptr;
    }

    try {
        auto* context = new VisualSimulationOfSmokeContext{new visual_smoke::Solver(*desc), {}};
        visual_smoke::g_last_error.clear();
        return context;
    } catch (const std::exception& ex) {
        set_global_error(ex.what());
        return nullptr;
    } catch (...) {
        set_global_error("unknown visual-simulation-of-smoke exception");
        return nullptr;
    }
}

void visual_simulation_of_smoke_context_destroy(VisualSimulationOfSmokeContext* context) {
    if (context == nullptr) {
        return;
    }
    delete context->solver;
    delete context;
}

uint64_t visual_simulation_of_smoke_context_required_density_bytes(const VisualSimulationOfSmokeContext* context) {
    return context != nullptr && context->solver != nullptr ? context->solver->density_bytes() : 0;
}

uint64_t visual_simulation_of_smoke_context_required_temperature_bytes(const VisualSimulationOfSmokeContext* context) {
    return context != nullptr && context->solver != nullptr ? context->solver->temperature_bytes() : 0;
}

int32_t visual_simulation_of_smoke_clear_async(VisualSimulationOfSmokeContext* context, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_clear_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->clear(to_stream(cuda_stream)); });
}

int32_t visual_simulation_of_smoke_add_source_async(VisualSimulationOfSmokeContext* context, const VisualSimulationOfSmokeSourceDesc* source, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr || source == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_add_source_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->add_source(*source, to_stream(cuda_stream)); });
}

int32_t visual_simulation_of_smoke_step_async(VisualSimulationOfSmokeContext* context, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_step_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->step(to_stream(cuda_stream)); });
}

int32_t visual_simulation_of_smoke_snapshot_density_async(VisualSimulationOfSmokeContext* context, VisualSimulationOfSmokeBufferView destination, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_snapshot_density_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->snapshot_density(destination, to_stream(cuda_stream)); });
}

int32_t visual_simulation_of_smoke_snapshot_temperature_async(VisualSimulationOfSmokeContext* context, VisualSimulationOfSmokeBufferView destination, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_snapshot_temperature_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->snapshot_temperature(destination, to_stream(cuda_stream)); });
}

int32_t visual_simulation_of_smoke_snapshot_velocity_magnitude_async(VisualSimulationOfSmokeContext* context, VisualSimulationOfSmokeBufferView destination, void* cuda_stream) {
    if (context == nullptr || context->solver == nullptr) {
        return store_error(context, VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT, "visual_simulation_of_smoke_snapshot_velocity_magnitude_async received a null argument");
    }
    return smoke_try(context, [&] { context->solver->snapshot_velocity_magnitude(destination, to_stream(cuda_stream)); });
}

uint64_t visual_simulation_of_smoke_last_error_length(void) {
    return static_cast<uint64_t>(visual_smoke::g_last_error.size());
}

uint64_t visual_simulation_of_smoke_context_last_error_length(const VisualSimulationOfSmokeContext* context) {
    return context != nullptr ? static_cast<uint64_t>(context->last_error.size()) : static_cast<uint64_t>(visual_smoke::g_last_error.size());
}

int32_t visual_simulation_of_smoke_copy_last_error(char* buffer, uint64_t buffer_size) {
    return copy_error_text(visual_smoke::g_last_error, buffer, buffer_size);
}

int32_t visual_simulation_of_smoke_copy_context_last_error(const VisualSimulationOfSmokeContext* context, char* buffer, uint64_t buffer_size) {
    return copy_error_text(context != nullptr ? context->last_error : visual_smoke::g_last_error, buffer, buffer_size);
}

} // extern "C"
