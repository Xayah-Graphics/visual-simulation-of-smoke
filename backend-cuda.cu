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

        __host__ __device__ inline int clampi(int value, int lo, int hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ inline float clampf(float value, float lo, float hi) {
            return value < lo ? lo : (value > hi ? hi : value);
        }

        __host__ __device__ inline std::uint64_t index_3d(int x, int y, int z, int sx, int sy) {
            return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
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
                const int x0   = clampi(static_cast<int>(floorf(gx)), 0, sx - 1);
                const int y0   = clampi(static_cast<int>(floorf(gy)), 0, sy - 1);
                const int z0   = clampi(static_cast<int>(floorf(gz)), 0, sz - 1);
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
            return make_float3(clampf(pos.x, 0.0f, static_cast<float>(nx) * h), clampf(pos.y, 0.0f, static_cast<float>(ny) * h), clampf(pos.z, 0.0f, static_cast<float>(nz) * h));
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

        __global__ void set_u_boundary_kernel(float* u, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i <= nx && j < ny && k < nz && (i == 0 || i == nx)) u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
        }

        __global__ void set_v_boundary_kernel(float* v, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j <= ny && k < nz && (j == 0 || j == ny)) v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
        }

        __global__ void set_w_boundary_kernel(float* w, int nx, int ny, int nz) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k <= nz && (k == 0 || k == nz)) w[index_3d(i, j, k, nx, ny)] = 0.0f;
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

        __global__ void apply_u_forces_kernel(float* u, const float* force_x, int nx, int ny, int nz, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > 0 && i < nx && j < ny && k < nz) u[index_3d(i, j, k, nx + 1, ny)] += 0.5f * dt * (force_x[index_3d(i - 1, j, k, nx, ny)] + force_x[index_3d(i, j, k, nx, ny)]);
        }

        __global__ void apply_v_forces_kernel(float* v, const float* density, const float* temperature, const float* force_y, int nx, int ny, int nz, float ambient_temperature, float density_buoyancy, float temperature_buoyancy, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j <= 0 || j >= ny || k >= nz) return;

            const std::uint64_t below   = index_3d(i, j - 1, k, nx, ny);
            const std::uint64_t above   = index_3d(i, j, k, nx, ny);
            const float density_avg     = 0.5f * (density[below] + density[above]);
            const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
            const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
            const float buoyancy        = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
            v[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
        }

        __global__ void apply_w_forces_kernel(float* w, const float* force_z, int nx, int ny, int nz, float dt) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k <= 0 || k >= nz) return;
            w[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
        }

        __global__ void advect_u_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i > nx || j >= ny || k >= nz) return;
            const float3 pos                   = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
            const float3 vel                   = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
            dst[index_3d(i, j, k, nx + 1, ny)] = sample_u(src_u, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
        }

        __global__ void advect_v_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j > ny || k >= nz) return;
            const float3 pos                   = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
            const float3 vel                   = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
            dst[index_3d(i, j, k, nx, ny + 1)] = sample_v(src_v, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
        }

        __global__ void advect_w_kernel(float* dst, const float* src_u, const float* src_v, const float* src_w, int nx, int ny, int nz, float h, float dt, bool cubic) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i >= nx || j >= ny || k > nz) return;
            const float3 pos               = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
            const float3 vel               = sample_velocity(src_u, src_v, src_w, pos, nx, ny, nz, h, cubic);
            dst[index_3d(i, j, k, nx, ny)] = sample_w(src_w, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
        }

        __global__ void compute_divergence_kernel(float* divergence, const float* u, const float* v, const float* w, int nx, int ny, int nz, float h) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
            if (i < nx && j < ny && k < nz) {
                divergence[index_3d(i, j, k, nx, ny)] = (fetch_clamped(u, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(u, i, j, k, nx + 1, ny, nz) + fetch_clamped(v, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(v, i, j, k, nx, ny + 1, nz) + fetch_clamped(w, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(w, i, j, k, nx, ny, nz + 1)) / h;
            }
        }

        __global__ void pressure_rbgs_kernel(float* pressure, const float* divergence, int nx, int ny, int nz, float h, float dt, int parity) {
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
                const float3 pos               = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
                const float3 vel               = sample_velocity(u, v, w, pos, nx, ny, nz, h, cubic);
                float value                    = sample_scalar(src, clamp_domain(make_float3(pos.x - dt * vel.x, pos.y - dt * vel.y, pos.z - dt * vel.z), nx, ny, nz, h), nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx, ny)] = clamp_nonnegative ? fmaxf(0.0f, value) : value;
            }
        }

    } // namespace

} // namespace visual_smoke

namespace {

    [[nodiscard]] visual_smoke::Stream to_stream(void* cuda_stream) noexcept {
        return reinterpret_cast<visual_smoke::Stream>(cuda_stream);
    }

} // namespace

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

    auto* density_prev     = reinterpret_cast<float*>(desc->temporary_previous_density);
    auto* temperature_prev = reinterpret_cast<float*>(desc->temporary_previous_temperature);
    auto* u_prev           = reinterpret_cast<float*>(desc->temporary_previous_velocity_x);
    auto* v_prev           = reinterpret_cast<float*>(desc->temporary_previous_velocity_y);
    auto* w_prev           = reinterpret_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure         = reinterpret_cast<float*>(desc->temporary_pressure);
    auto* divergence       = reinterpret_cast<float*>(desc->temporary_divergence);
    auto* omega_x          = reinterpret_cast<float*>(desc->temporary_omega_x);
    auto* omega_y          = reinterpret_cast<float*>(desc->temporary_omega_y);
    auto* omega_z          = reinterpret_cast<float*>(desc->temporary_omega_z);
    auto* omega_mag        = reinterpret_cast<float*>(desc->temporary_omega_magnitude);
    auto* force_x          = reinterpret_cast<float*>(desc->temporary_force_x);
    auto* force_y          = reinterpret_cast<float*>(desc->temporary_force_y);
    auto* force_z          = reinterpret_cast<float*>(desc->temporary_force_z);
    auto* density_f        = reinterpret_cast<float*>(desc->density);
    auto* temperature_f    = reinterpret_cast<float*>(desc->temperature);
    auto* u                = reinterpret_cast<float*>(desc->velocity_x);
    auto* v                = reinterpret_cast<float*>(desc->velocity_y);
    auto* w                = reinterpret_cast<float*>(desc->velocity_z);
    const dim3 block{static_cast<unsigned>(std::max(block_x, 1)), static_cast<unsigned>(std::max(block_y, 1)), static_cast<unsigned>(std::max(block_z, 1))};
    const dim3 cells  = make_grid(nx, ny, nz, block);
    const dim3 u_grid = make_grid(nx + 1, ny, nz, block);
    const dim3 v_grid = make_grid(nx, ny + 1, nz, block);
    const dim3 w_grid = make_grid(nx, ny, nz + 1, block);
    const bool cubic  = use_monotonic_cubic != 0u;
    const auto stream = to_stream(desc->stream);

    nvtx3::scoped_range step_range{"vsmoke.step"};
    {
        nvtx3::scoped_range range{"vsmoke.step.forces"};
        compute_vorticity_kernel<<<cells, block, 0, stream>>>(u, v, w, omega_x, omega_y, omega_z, omega_mag, nx, ny, nz, cell_size);
        compute_confinement_kernel<<<cells, block, 0, stream>>>(omega_x, omega_y, omega_z, omega_mag, force_x, force_y, force_z, nx, ny, nz, vorticity_epsilon, cell_size);
        apply_u_forces_kernel<<<u_grid, block, 0, stream>>>(u, force_x, nx, ny, nz, dt);
        apply_v_forces_kernel<<<v_grid, block, 0, stream>>>(v, density_f, temperature_f, force_y, nx, ny, nz, ambient_temperature, density_buoyancy, temperature_buoyancy, dt);
        apply_w_forces_kernel<<<w_grid, block, 0, stream>>>(w, force_z, nx, ny, nz, dt);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"vsmoke.step.advect_velocity"};
        advect_u_kernel<<<u_grid, block, 0, stream>>>(u_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic);
        advect_v_kernel<<<v_grid, block, 0, stream>>>(v_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic);
        advect_w_kernel<<<w_grid, block, 0, stream>>>(w_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic);
        set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(u_prev, nx, ny, nz);
        set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(v_prev, nx, ny, nz);
        set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(w_prev, nx, ny, nz);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    {
        nvtx3::scoped_range range{"vsmoke.step.project"};
        if (cuda_code(cudaMemsetAsync(pressure, 0, cell_bytes, stream)) != 0) return 5001;
        compute_divergence_kernel<<<cells, block, 0, stream>>>(divergence, u_prev, v_prev, w_prev, nx, ny, nz, cell_size);
        for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
            pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure, divergence, nx, ny, nz, cell_size, dt, 0);
            pressure_rbgs_kernel<<<cells, block, 0, stream>>>(pressure, divergence, nx, ny, nz, cell_size, dt, 1);
        }
        subtract_gradient_u_kernel<<<u_grid, block, 0, stream>>>(u_prev, pressure, nx, ny, nz, cell_size, dt);
        subtract_gradient_v_kernel<<<v_grid, block, 0, stream>>>(v_prev, pressure, nx, ny, nz, cell_size, dt);
        subtract_gradient_w_kernel<<<w_grid, block, 0, stream>>>(w_prev, pressure, nx, ny, nz, cell_size, dt);
        set_u_boundary_kernel<<<u_grid, block, 0, stream>>>(u_prev, nx, ny, nz);
        set_v_boundary_kernel<<<v_grid, block, 0, stream>>>(v_prev, nx, ny, nz);
        set_w_boundary_kernel<<<w_grid, block, 0, stream>>>(w_prev, nx, ny, nz);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    if (cuda_code(cudaMemcpyAsync(u, u_prev, u_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(v, v_prev, v_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(w, w_prev, w_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(density_prev, density_f, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    if (cuda_code(cudaMemcpyAsync(temperature_prev, temperature_f, cell_bytes, cudaMemcpyDeviceToDevice, stream)) != 0) return 5001;
    {
        nvtx3::scoped_range range{"vsmoke.step.advect_scalars"};
        advect_scalar_kernel<<<cells, block, 0, stream>>>(density_f, density_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic, true);
        advect_scalar_kernel<<<cells, block, 0, stream>>>(temperature_f, temperature_prev, u, v, w, nx, ny, nz, cell_size, dt, cubic, false);
        if (cuda_code(cudaGetLastError()) != 0) return 5001;
    }
    return 0;
}

} // extern "C"
