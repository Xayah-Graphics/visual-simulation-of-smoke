#include "visual-simulation-of-smoke.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

int clampi(const int value, const int lo, const int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

float clampf(const float value, const float lo, const float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

std::uint64_t index_3d(const int x, const int y, const int z, const int sx, const int sy) {
    return static_cast<std::uint64_t>(z) * static_cast<std::uint64_t>(sx) * static_cast<std::uint64_t>(sy) + static_cast<std::uint64_t>(y) * static_cast<std::uint64_t>(sx) + static_cast<std::uint64_t>(x);
}

float fetch_clamped(const float* field, const int x, const int y, const int z, const int sx, const int sy, const int sz) {
    return field[index_3d(clampi(x, 0, sx - 1), clampi(y, 0, sy - 1), clampi(z, 0, sz - 1), sx, sy)];
}

float monotonic_cubic(const float p0, const float p1, const float p2, const float p3, const float t) {
    const float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    const float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    const float a2 = -0.5f * p0 + 0.5f * p2;
    const float a3 = p1;
    return clampf(((a0 * t + a1) * t + a2) * t + a3, std::fmin(p1, p2), std::fmax(p1, p2));
}

float sample_grid(const float* field, float gx, float gy, float gz, const int sx, const int sy, const int sz, const bool cubic) {
    gx = clampf(gx, 0.0f, static_cast<float>(sx - 1));
    gy = clampf(gy, 0.0f, static_cast<float>(sy - 1));
    gz = clampf(gz, 0.0f, static_cast<float>(sz - 1));
    if (!cubic) {
        const int x0 = clampi(static_cast<int>(std::floor(gx)), 0, sx - 1);
        const int y0 = clampi(static_cast<int>(std::floor(gy)), 0, sy - 1);
        const int z0 = clampi(static_cast<int>(std::floor(gz)), 0, sz - 1);
        const int x1 = std::min(x0 + 1, sx - 1);
        const int y1 = std::min(y0 + 1, sy - 1);
        const int z1 = std::min(z0 + 1, sz - 1);
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

    const int ix = static_cast<int>(std::floor(gx));
    const int iy = static_cast<int>(std::floor(gy));
    const int iz = static_cast<int>(std::floor(gz));
    const float tx = gx - static_cast<float>(ix);
    const float ty = gy - static_cast<float>(iy);
    const float tz = gz - static_cast<float>(iz);
    float yz[4][4];
    for (int zz = 0; zz < 4; ++zz)
        for (int yy = 0; yy < 4; ++yy)
            yz[zz][yy] = monotonic_cubic(fetch_clamped(field, ix - 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 0, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 1, iy + yy - 1, iz + zz - 1, sx, sy, sz), fetch_clamped(field, ix + 2, iy + yy - 1, iz + zz - 1, sx, sy, sz), tx);
    float zline[4];
    for (int zz = 0; zz < 4; ++zz) zline[zz] = monotonic_cubic(yz[zz][0], yz[zz][1], yz[zz][2], yz[zz][3], ty);
    return monotonic_cubic(zline[0], zline[1], zline[2], zline[3], tz);
}

float sample_scalar(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const bool cubic) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h - 0.5f, nx, ny, nz, cubic);
}

float sample_u(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const bool cubic) {
    return sample_grid(field, x / h, y / h - 0.5f, z / h - 0.5f, nx + 1, ny, nz, cubic);
}

float sample_v(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const bool cubic) {
    return sample_grid(field, x / h - 0.5f, y / h, z / h - 0.5f, nx, ny + 1, nz, cubic);
}

float sample_w(const float* field, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const bool cubic) {
    return sample_grid(field, x / h - 0.5f, y / h - 0.5f, z / h, nx, ny, nz + 1, cubic);
}

void clamp_domain(float& x, float& y, float& z, const int nx, const int ny, const int nz, const float h) {
    x = clampf(x, 0.0f, static_cast<float>(nx) * h);
    y = clampf(y, 0.0f, static_cast<float>(ny) * h);
    z = clampf(z, 0.0f, static_cast<float>(nz) * h);
}

void sample_velocity(const float* u, const float* v, const float* w, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const bool cubic, float& out_x, float& out_y, float& out_z) {
    clamp_domain(x, y, z, nx, ny, nz, h);
    out_x = sample_u(u, x, y, z, nx, ny, nz, h, cubic);
    out_y = sample_v(v, x, y, z, nx, ny, nz, h, cubic);
    out_z = sample_w(w, x, y, z, nx, ny, nz, h, cubic);
}

float center_u(const float* u, const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    const int ci = clampi(i, 0, nx - 1);
    const int cj = clampi(j, 0, ny - 1);
    const int ck = clampi(k, 0, nz - 1);
    return 0.5f * (fetch_clamped(u, ci, cj, ck, nx + 1, ny, nz) + fetch_clamped(u, ci + 1, cj, ck, nx + 1, ny, nz));
}

float center_v(const float* v, const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    const int ci = clampi(i, 0, nx - 1);
    const int cj = clampi(j, 0, ny - 1);
    const int ck = clampi(k, 0, nz - 1);
    return 0.5f * (fetch_clamped(v, ci, cj, ck, nx, ny + 1, nz) + fetch_clamped(v, ci, cj + 1, ck, nx, ny + 1, nz));
}

float center_w(const float* w, const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    const int ci = clampi(i, 0, nx - 1);
    const int cj = clampi(j, 0, ny - 1);
    const int ck = clampi(k, 0, nz - 1);
    return 0.5f * (fetch_clamped(w, ci, cj, ck, nx, ny, nz + 1) + fetch_clamped(w, ci, cj, ck + 1, nx, ny, nz + 1));
}

void set_u_boundary(float* u, const int nx, const int ny, const int nz) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            u[index_3d(0, j, k, nx + 1, ny)] = 0.0f;
            u[index_3d(nx, j, k, nx + 1, ny)] = 0.0f;
        }
}

void set_v_boundary(float* v, const int nx, const int ny, const int nz) {
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            v[index_3d(i, 0, k, nx, ny + 1)] = 0.0f;
            v[index_3d(i, ny, k, nx, ny + 1)] = 0.0f;
        }
}

void set_w_boundary(float* w, const int nx, const int ny, const int nz) {
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            w[index_3d(i, j, 0, nx, ny)] = 0.0f;
            w[index_3d(i, j, nz, nx, ny)] = 0.0f;
        }
}

void compute_vorticity(const float* u, const float* v, const float* w, float* omega_x, float* omega_y, float* omega_z, float* omega_mag, const int nx, const int ny, const int nz, const float h) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const float dw_dy = (center_w(w, i, j + 1, k, nx, ny, nz) - center_w(w, i, j - 1, k, nx, ny, nz)) / (2.0f * h);
                const float dv_dz = (center_v(v, i, j, k + 1, nx, ny, nz) - center_v(v, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
                const float du_dz = (center_u(u, i, j, k + 1, nx, ny, nz) - center_u(u, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
                const float dw_dx = (center_w(w, i + 1, j, k, nx, ny, nz) - center_w(w, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
                const float dv_dx = (center_v(v, i + 1, j, k, nx, ny, nz) - center_v(v, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
                const float du_dy = (center_u(u, i, j + 1, k, nx, ny, nz) - center_u(u, i, j - 1, k, nx, ny, nz)) / (2.0f * h);
                const float wx = dw_dy - dv_dz;
                const float wy = du_dz - dw_dx;
                const float wz = dv_dx - du_dy;
                const auto index = index_3d(i, j, k, nx, ny);
                omega_x[index] = wx;
                omega_y[index] = wy;
                omega_z[index] = wz;
                omega_mag[index] = std::sqrt(wx * wx + wy * wy + wz * wz);
            }
}

void compute_confinement(const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_mag, float* force_x, float* force_y, float* force_z, const int nx, const int ny, const int nz, const float epsilon, const float h) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const float gx = (fetch_clamped(omega_mag, i + 1, j, k, nx, ny, nz) - fetch_clamped(omega_mag, i - 1, j, k, nx, ny, nz)) / (2.0f * h);
                const float gy = (fetch_clamped(omega_mag, i, j + 1, k, nx, ny, nz) - fetch_clamped(omega_mag, i, j - 1, k, nx, ny, nz)) / (2.0f * h);
                const float gz = (fetch_clamped(omega_mag, i, j, k + 1, nx, ny, nz) - fetch_clamped(omega_mag, i, j, k - 1, nx, ny, nz)) / (2.0f * h);
                const float inv_len = 1.0f / std::sqrt(std::max(gx * gx + gy * gy + gz * gz, 1.0e-12f));
                const float nxv = gx * inv_len;
                const float nyv = gy * inv_len;
                const float nzv = gz * inv_len;
                const auto index = index_3d(i, j, k, nx, ny);
                force_x[index] = epsilon * h * (nyv * omega_z[index] - nzv * omega_y[index]);
                force_y[index] = epsilon * h * (nzv * omega_x[index] - nxv * omega_z[index]);
                force_z[index] = epsilon * h * (nxv * omega_y[index] - nyv * omega_x[index]);
            }
}

void apply_u_forces(float* u, const float* force_x, const int nx, const int ny, const int nz, const float dt) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i < nx; ++i)
                u[index_3d(i, j, k, nx + 1, ny)] += 0.5f * dt * (force_x[index_3d(i - 1, j, k, nx, ny)] + force_x[index_3d(i, j, k, nx, ny)]);
}

void apply_v_forces(float* v, const float* density, const float* temperature, const float* force_y, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_buoyancy, const float temperature_buoyancy, const float dt) {
    for (int k = 0; k < nz; ++k)
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const auto below = index_3d(i, j - 1, k, nx, ny);
                const auto above = index_3d(i, j, k, nx, ny);
                const float density_avg = 0.5f * (density[below] + density[above]);
                const float temperature_avg = 0.5f * (temperature[below] + temperature[above]);
                const float confinement_avg = 0.5f * (force_y[below] + force_y[above]);
                const float buoyancy = temperature_buoyancy * (temperature_avg - ambient_temperature) - density_buoyancy * density_avg;
                v[index_3d(i, j, k, nx, ny + 1)] += dt * (buoyancy + confinement_avg);
            }
}

void apply_w_forces(float* w, const float* force_z, const int nx, const int ny, const int nz, const float dt) {
    for (int k = 1; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                w[index_3d(i, j, k, nx, ny)] += 0.5f * dt * (force_z[index_3d(i, j, k - 1, nx, ny)] + force_z[index_3d(i, j, k, nx, ny)]);
}

void advect_u(float* dst, const float* src_u, const float* src_v, const float* src_w, const int nx, const int ny, const int nz, const float h, const float dt, const bool cubic) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                float px = static_cast<float>(i) * h;
                float py = (static_cast<float>(j) + 0.5f) * h;
                float pz = (static_cast<float>(k) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(src_u, src_v, src_w, px, py, pz, nx, ny, nz, h, cubic, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                dst[index_3d(i, j, k, nx + 1, ny)] = sample_u(src_u, px, py, pz, nx, ny, nz, h, cubic);
            }
}

void advect_v(float* dst, const float* src_u, const float* src_v, const float* src_w, const int nx, const int ny, const int nz, const float h, const float dt, const bool cubic) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * h;
                float py = static_cast<float>(j) * h;
                float pz = (static_cast<float>(k) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(src_u, src_v, src_w, px, py, pz, nx, ny, nz, h, cubic, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                dst[index_3d(i, j, k, nx, ny + 1)] = sample_v(src_v, px, py, pz, nx, ny, nz, h, cubic);
            }
}

void advect_w(float* dst, const float* src_u, const float* src_v, const float* src_w, const int nx, const int ny, const int nz, const float h, const float dt, const bool cubic) {
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * h;
                float py = (static_cast<float>(j) + 0.5f) * h;
                float pz = static_cast<float>(k) * h;
                float vx, vy, vz;
                sample_velocity(src_u, src_v, src_w, px, py, pz, nx, ny, nz, h, cubic, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                dst[index_3d(i, j, k, nx, ny)] = sample_w(src_w, px, py, pz, nx, ny, nz, h, cubic);
            }
}

void compute_divergence(float* divergence, const float* u, const float* v, const float* w, const int nx, const int ny, const int nz, const float h) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                divergence[index_3d(i, j, k, nx, ny)] = (fetch_clamped(u, i + 1, j, k, nx + 1, ny, nz) - fetch_clamped(u, i, j, k, nx + 1, ny, nz)
                    + fetch_clamped(v, i, j + 1, k, nx, ny + 1, nz) - fetch_clamped(v, i, j, k, nx, ny + 1, nz) + fetch_clamped(w, i, j, k + 1, nx, ny, nz + 1) - fetch_clamped(w, i, j, k, nx, ny, nz + 1)) / h;
}

void pressure_rbgs(float* pressure, const float* divergence, const int nx, const int ny, const int nz, const float h, const float dt, const int parity) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (((i + j + k) & 1) != parity) continue;
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
}

void subtract_gradient_u(float* u, const float* pressure, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 1; i < nx; ++i)
                u[index_3d(i, j, k, nx + 1, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) / h;
}

void subtract_gradient_v(float* v, const float* pressure, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int k = 0; k < nz; ++k)
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                v[index_3d(i, j, k, nx, ny + 1)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) / h;
}

void subtract_gradient_w(float* w, const float* pressure, const int nx, const int ny, const int nz, const float h, const float dt) {
    for (int k = 1; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                w[index_3d(i, j, k, nx, ny)] -= dt * (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) / h;
}

void advect_scalar(float* dst, const float* src, const float* u, const float* v, const float* w, const int nx, const int ny, const int nz, const float h, const float dt, const bool cubic, const bool clamp_nonnegative) {
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float px = (static_cast<float>(i) + 0.5f) * h;
                float py = (static_cast<float>(j) + 0.5f) * h;
                float pz = (static_cast<float>(k) + 0.5f) * h;
                float vx, vy, vz;
                sample_velocity(u, v, w, px, py, pz, nx, ny, nz, h, cubic, vx, vy, vz);
                px -= dt * vx;
                py -= dt * vy;
                pz -= dt * vz;
                clamp_domain(px, py, pz, nx, ny, nz, h);
                float value = sample_scalar(src, px, py, pz, nx, ny, nz, h, cubic);
                dst[index_3d(i, j, k, nx, ny)] = clamp_nonnegative ? std::max(0.0f, value) : value;
            }
}

} // namespace

extern "C" {

int32_t visual_simulation_of_smoke_step_cpu(const VisualSimulationOfSmokeStepDesc* desc) {
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
    const bool cubic = desc->use_monotonic_cubic != 0u;

    auto* density = reinterpret_cast<float*>(desc->density);
    auto* temperature = reinterpret_cast<float*>(desc->temperature);
    auto* velocity_x = reinterpret_cast<float*>(desc->velocity_x);
    auto* velocity_y = reinterpret_cast<float*>(desc->velocity_y);
    auto* velocity_z = reinterpret_cast<float*>(desc->velocity_z);
    auto* previous_density = reinterpret_cast<float*>(desc->temporary_previous_density);
    auto* previous_temperature = reinterpret_cast<float*>(desc->temporary_previous_temperature);
    auto* previous_velocity_x = reinterpret_cast<float*>(desc->temporary_previous_velocity_x);
    auto* previous_velocity_y = reinterpret_cast<float*>(desc->temporary_previous_velocity_y);
    auto* previous_velocity_z = reinterpret_cast<float*>(desc->temporary_previous_velocity_z);
    auto* pressure = reinterpret_cast<float*>(desc->temporary_pressure);
    auto* divergence = reinterpret_cast<float*>(desc->temporary_divergence);
    auto* omega_x = reinterpret_cast<float*>(desc->temporary_omega_x);
    auto* omega_y = reinterpret_cast<float*>(desc->temporary_omega_y);
    auto* omega_z = reinterpret_cast<float*>(desc->temporary_omega_z);
    auto* omega_magnitude = reinterpret_cast<float*>(desc->temporary_omega_magnitude);
    auto* force_x = reinterpret_cast<float*>(desc->temporary_force_x);
    auto* force_y = reinterpret_cast<float*>(desc->temporary_force_y);
    auto* force_z = reinterpret_cast<float*>(desc->temporary_force_z);

    const std::uint64_t cell_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_x_field_bytes = static_cast<std::uint64_t>(nx + 1) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_y_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny + 1) * static_cast<std::uint64_t>(nz) * sizeof(float);
    const std::uint64_t velocity_z_field_bytes = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz + 1) * sizeof(float);

    compute_vorticity(velocity_x, velocity_y, velocity_z, omega_x, omega_y, omega_z, omega_magnitude, nx, ny, nz, cell_size);
    compute_confinement(omega_x, omega_y, omega_z, omega_magnitude, force_x, force_y, force_z, nx, ny, nz, vorticity_epsilon, cell_size);
    apply_u_forces(velocity_x, force_x, nx, ny, nz, dt);
    apply_v_forces(velocity_y, density, temperature, force_y, nx, ny, nz, ambient_temperature, density_buoyancy, temperature_buoyancy, dt);
    apply_w_forces(velocity_z, force_z, nx, ny, nz, dt);

    advect_u(previous_velocity_x, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, dt, cubic);
    advect_v(previous_velocity_y, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, dt, cubic);
    advect_w(previous_velocity_z, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, dt, cubic);
    set_u_boundary(previous_velocity_x, nx, ny, nz);
    set_v_boundary(previous_velocity_y, nx, ny, nz);
    set_w_boundary(previous_velocity_z, nx, ny, nz);

    std::memset(pressure, 0, static_cast<std::size_t>(cell_bytes));
    compute_divergence(divergence, previous_velocity_x, previous_velocity_y, previous_velocity_z, nx, ny, nz, cell_size);
    for (int iteration = 0; iteration < pressure_iterations; ++iteration) {
        pressure_rbgs(pressure, divergence, nx, ny, nz, cell_size, dt, 0);
        pressure_rbgs(pressure, divergence, nx, ny, nz, cell_size, dt, 1);
    }
    subtract_gradient_u(previous_velocity_x, pressure, nx, ny, nz, cell_size, dt);
    subtract_gradient_v(previous_velocity_y, pressure, nx, ny, nz, cell_size, dt);
    subtract_gradient_w(previous_velocity_z, pressure, nx, ny, nz, cell_size, dt);
    set_u_boundary(previous_velocity_x, nx, ny, nz);
    set_v_boundary(previous_velocity_y, nx, ny, nz);
    set_w_boundary(previous_velocity_z, nx, ny, nz);

    std::memcpy(velocity_x, previous_velocity_x, velocity_x_field_bytes);
    std::memcpy(velocity_y, previous_velocity_y, velocity_y_field_bytes);
    std::memcpy(velocity_z, previous_velocity_z, velocity_z_field_bytes);
    std::memcpy(previous_density, density, static_cast<std::size_t>(cell_bytes));
    std::memcpy(previous_temperature, temperature, static_cast<std::size_t>(cell_bytes));
    advect_scalar(density, previous_density, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, dt, cubic, true);
    advect_scalar(temperature, previous_temperature, velocity_x, velocity_y, velocity_z, nx, ny, nz, cell_size, dt, cubic, false);

    return 0;
}

} // extern "C"
