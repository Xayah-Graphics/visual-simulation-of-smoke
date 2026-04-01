#include "visual-simulation-of-smoke-3d.h"
#include <algorithm>
#include <array>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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

        struct StepGraphStorage {
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
        } step_graph{};

        struct DeviceBuffers {
            struct Flow {
                float* velocity_x                   = nullptr;
                float* velocity_y                   = nullptr;
                float* velocity_z                   = nullptr;
                float* temp_velocity_x              = nullptr;
                float* temp_velocity_y              = nullptr;
                float* temp_velocity_z              = nullptr;
                float* centered_velocity_x          = nullptr;
                float* centered_velocity_y          = nullptr;
                float* centered_velocity_z          = nullptr;
                float* velocity_magnitude           = nullptr;
                float* pressure                     = nullptr;
                float* pressure_rhs                 = nullptr;
                float* divergence                   = nullptr;
                float* vorticity_x                  = nullptr;
                float* vorticity_y                  = nullptr;
                float* vorticity_z                  = nullptr;
                float* vorticity_magnitude          = nullptr;
                float* force_x                      = nullptr;
                float* force_y                      = nullptr;
                float* force_z                      = nullptr;
                int* pressure_anchor                = nullptr;
                int* pressure_row_offsets           = nullptr;
                int* pressure_column_indices        = nullptr;
                float* pressure_values              = nullptr;
                int* pressure_factor_row_offsets    = nullptr;
                int* pressure_factor_column_indices = nullptr;
                float* pressure_factor_values       = nullptr;
                float* pcg_r                        = nullptr;
                float* pcg_p                        = nullptr;
                float* pcg_ap                       = nullptr;
                float* pcg_z                        = nullptr;
                float* pcg_y                        = nullptr;
                float* pressure_dot_rz              = nullptr;
                float* pressure_dot_pap             = nullptr;
                float* pressure_dot_rr              = nullptr;
                float* pressure_alpha               = nullptr;
                float* pressure_negative_alpha      = nullptr;
                float* pressure_beta                = nullptr;
                float* pressure_one                 = nullptr;
                int pressure_nnz                    = 0;
                int pressure_factor_nnz             = 0;
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

        struct PressureSolverStorage {
            cublasHandle_t cublas               = nullptr;
            cusparseHandle_t cusparse           = nullptr;
            cusparseMatDescr_t factor_descr     = nullptr;
            csric02Info_t factor_info           = nullptr;
            cusparseSpMatDescr_t matrix         = nullptr;
            cusparseSpMatDescr_t factor         = nullptr;
            cusparseDnVecDescr_t vec_r          = nullptr;
            cusparseDnVecDescr_t vec_p          = nullptr;
            cusparseDnVecDescr_t vec_ap         = nullptr;
            cusparseDnVecDescr_t vec_y          = nullptr;
            cusparseDnVecDescr_t vec_z          = nullptr;
            cusparseSpSVDescr_t lower_solve     = nullptr;
            cusparseSpSVDescr_t upper_solve     = nullptr;
            void* factor_buffer                 = nullptr;
            std::size_t factor_buffer_size      = 0;
            void* spmv_buffer                   = nullptr;
            std::size_t spmv_buffer_size        = 0;
            void* lower_solve_buffer            = nullptr;
            std::size_t lower_solve_buffer_size = 0;
            void* upper_solve_buffer            = nullptr;
            std::size_t upper_solve_buffer_size = 0;
        } pressure_solver{};
    };

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

    __host__ __device__ int wrap_index(int value, const int size) {
        if (size <= 0) return 0;
        value %= size;
        if (value < 0) value += size;
        return value;
    }

    __host__ __device__ bool cell_in_bounds(const int x, const int y, const int z, const int nx, const int ny, const int nz) {
        return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
    }

    __host__ __device__ bool resolve_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) x = wrap_index(x, nx);
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) y = wrap_index(y, ny);
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __host__ __device__ bool resolve_scalar_cell_coordinates(int& x, int& y, int& z, const int nx, const int ny, const int nz, const SmokeSimulationScalarBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nx > 0) x = wrap_index(x, nx);
        if (boundary.y_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && ny > 0) y = wrap_index(y, ny);
        if (boundary.z_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nz > 0) z = wrap_index(z, nz);
        return cell_in_bounds(x, y, z, nx, ny, nz);
    }

    __device__ bool load_occupancy(const uint8_t* occupancy, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (occupancy == nullptr) return false;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return true;
        return occupancy[index_3d(x, y, z, nx, ny)] != 0;
    }

    __device__ float load_scalar(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationScalarBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            const auto [type, value] = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nx > 0) {
                x = wrap_index(x, nx);
            } else if (type == SMOKE_SIMULATION_SCALAR_BOUNDARY_ZERO_FLUX && nx > 0) {
                x = x < 0 ? 0 : nx - 1;
            } else {
                return value;
            }
        }
        if (y < 0 || y >= ny) {
            const auto [type, value] = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && ny > 0) {
                y = wrap_index(y, ny);
            } else if (type == SMOKE_SIMULATION_SCALAR_BOUNDARY_ZERO_FLUX && ny > 0) {
                y = y < 0 ? 0 : ny - 1;
            } else {
                return value;
            }
        }
        if (z < 0 || z >= nz) {
            const auto [type, value] = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nz > 0) {
                z = wrap_index(z, nz);
            } else if (type == SMOKE_SIMULATION_SCALAR_BOUNDARY_ZERO_FLUX && nz > 0) {
                z = z < 0 ? 0 : nz - 1;
            } else {
                return value;
            }
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_flow_cell(const float* field, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                x = wrap_index(x, nx);
            } else {
                x = x < 0 ? 0 : nx - 1;
            }
        }
        if (y < 0 || y >= ny) {
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                y = wrap_index(y, ny);
            } else {
                y = y < 0 ? 0 : ny - 1;
            }
        }
        if (z < 0 || z >= nz) {
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                z = wrap_index(z, nz);
            } else {
                z = z < 0 ? 0 : nz - 1;
            }
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_center_velocity_component(const float* field, const int component_axis, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (x < 0 || x >= nx) {
            const auto face = x < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                x = wrap_index(x, nx);
            } else {
                const float interior = field[index_3d(x < 0 ? 0 : nx - 1, (std::clamp) (y, 0, ny - 1), (std::clamp) (z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 0) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (y < 0 || y >= ny) {
            const auto face = y < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                y = wrap_index(y, ny);
            } else {
                const float interior = field[index_3d((std::clamp) (x, 0, nx - 1), y < 0 ? 0 : ny - 1, (std::clamp) (z, 0, nz - 1), nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 1) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        if (z < 0 || z >= nz) {
            const auto face = z < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                z = wrap_index(z, nz);
            } else {
                const float interior = field[index_3d((std::clamp) (x, 0, nx - 1), (std::clamp) (y, 0, ny - 1), z < 0 ? 0 : nz - 1, nx, ny)];
                float prescribed     = 0.0f;
                if (component_axis == 0) prescribed = face.velocity_x;
                if (component_axis == 1) prescribed = face.velocity_y;
                if (component_axis == 2) prescribed = face.velocity_z;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL && component_axis != 2) return interior;
                return 2.0f * prescribed - interior;
            }
        }
        return field[index_3d(x, y, z, nx, ny)];
    }

    __device__ float load_velocity_x(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (i < 0 || i > nx) {
            const auto face = i < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                i = wrap_index(i, nx);
            } else {
                const float interior = field[index_velocity_x(i < 0 ? 0 : nx, (std::clamp) (j, 0, ny - 1), (std::clamp) (k, 0, nz - 1), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                return 2.0f * face.velocity_x - interior;
            }
        }
        if (j < 0 || j >= ny) {
            const auto face = j < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                j = wrap_index(j, ny);
            } else {
                const float interior = field[index_velocity_x((std::clamp) (i, 0, nx), j < 0 ? 0 : ny - 1, (std::clamp) (k, 0, nz - 1), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_x - interior;
            }
        }
        if (k < 0 || k >= nz) {
            const auto face = k < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                k = wrap_index(k, nz);
            } else {
                const float interior = field[index_velocity_x((std::clamp) (i, 0, nx), (std::clamp) (j, 0, ny - 1), k < 0 ? 0 : nz - 1, nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_x - interior;
            }
        }
        return field[index_velocity_x(i, j, k, nx, ny)];
    }

    __device__ float load_velocity_y(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (i < 0 || i >= nx) {
            const auto face = i < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                i = wrap_index(i, nx);
            } else {
                const float interior = field[index_velocity_y(i < 0 ? 0 : nx - 1, (std::clamp) (j, 0, ny), (std::clamp) (k, 0, nz - 1), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_y - interior;
            }
        }
        if (j < 0 || j > ny) {
            const auto face = j < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                j = wrap_index(j, ny);
            } else {
                const float interior = field[index_velocity_y((std::clamp) (i, 0, nx - 1), j < 0 ? 0 : ny, (std::clamp) (k, 0, nz - 1), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                return 2.0f * face.velocity_y - interior;
            }
        }
        if (k < 0 || k >= nz) {
            const auto face = k < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                k = wrap_index(k, nz);
            } else {
                const float interior = field[index_velocity_y((std::clamp) (i, 0, nx - 1), (std::clamp) (j, 0, ny), k < 0 ? 0 : nz - 1, nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_y - interior;
            }
        }
        return field[index_velocity_y(i, j, k, nx, ny)];
    }

    __device__ float load_velocity_z(const float* field, int i, int j, int k, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (i < 0 || i >= nx) {
            const auto face = i < 0 ? boundary.x_minus : boundary.x_plus;
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                i = wrap_index(i, nx);
            } else {
                const float interior = field[index_velocity_z(i < 0 ? 0 : nx - 1, (std::clamp) (j, 0, ny - 1), (std::clamp) (k, 0, nz), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_z - interior;
            }
        }
        if (j < 0 || j >= ny) {
            const auto face = j < 0 ? boundary.y_minus : boundary.y_plus;
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                j = wrap_index(j, ny);
            } else {
                const float interior = field[index_velocity_z((std::clamp) (i, 0, nx - 1), j < 0 ? 0 : ny - 1, (std::clamp) (k, 0, nz), nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL) return interior;
                return 2.0f * face.velocity_z - interior;
            }
        }
        if (k < 0 || k > nz) {
            const auto face = k < 0 ? boundary.z_minus : boundary.z_plus;
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
                k = wrap_index(k, nz);
            } else {
                const float interior = field[index_velocity_z((std::clamp) (i, 0, nx - 1), (std::clamp) (j, 0, ny - 1), k < 0 ? 0 : nz, nx, ny)];
                if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) return interior;
                return 2.0f * face.velocity_z - interior;
            }
        }
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

    __device__ float sample_scalar_linear(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationScalarBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nz > 0) {
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

    __device__ float sample_scalar_cubic(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationScalarBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC && nz > 0) {
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
                const int yy   = y1 + dy - 1;
                const int zz   = z1 + dz - 1;
                const float p0 = load_scalar(field, x1 - 1, yy, zz, nx, ny, nz, boundary);
                const float p1 = load_scalar(field, x1, yy, zz, nx, ny, nz, boundary);
                const float p2 = load_scalar(field, x1 + 1, yy, zz, nx, ny, nz, boundary);
                const float p3 = load_scalar(field, x1 + 2, yy, zz, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_velocity_x(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
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

    __device__ float sample_velocity_x_cubic(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }
        const float gx = x / h;
        const float gy = y / h - 0.5f;
        const float gz = z / h - 0.5f;
        const int i1   = static_cast<int>(floorf(gx));
        const int j1   = static_cast<int>(floorf(gy));
        const int k1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i1);
        const float ty = gy - static_cast<float>(j1);
        const float tz = gz - static_cast<float>(k1);
        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int jj   = j1 + dy - 1;
                const int kk   = k1 + dz - 1;
                const float p0 = load_velocity_x(field, i1 - 1, jj, kk, nx, ny, nz, boundary);
                const float p1 = load_velocity_x(field, i1, jj, kk, nx, ny, nz, boundary);
                const float p2 = load_velocity_x(field, i1 + 1, jj, kk, nx, ny, nz, boundary);
                const float p3 = load_velocity_x(field, i1 + 2, jj, kk, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_velocity_y(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
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

    __device__ float sample_velocity_y_cubic(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }
        const float gx = x / h - 0.5f;
        const float gy = y / h;
        const float gz = z / h - 0.5f;
        const int i1   = static_cast<int>(floorf(gx));
        const int j1   = static_cast<int>(floorf(gy));
        const int k1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i1);
        const float ty = gy - static_cast<float>(j1);
        const float tz = gz - static_cast<float>(k1);
        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int yy   = j1 + dy - 1;
                const int zz   = k1 + dz - 1;
                const float p0 = load_velocity_y(field, i1 - 1, yy, zz, nx, ny, nz, boundary);
                const float p1 = load_velocity_y(field, i1, yy, zz, nx, ny, nz, boundary);
                const float p2 = load_velocity_y(field, i1 + 1, yy, zz, nx, ny, nz, boundary);
                const float p3 = load_velocity_y(field, i1 + 2, yy, zz, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float sample_velocity_z(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
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

    __device__ float sample_velocity_z_cubic(const float* field, float x, float y, float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            x                    = fmodf(x, extent_x);
            if (x < 0.0f) x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            y                    = fmodf(y, extent_y);
            if (y < 0.0f) y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
            const float extent_z = static_cast<float>(nz) * h;
            z                    = fmodf(z, extent_z);
            if (z < 0.0f) z += extent_z;
        }
        const float gx = x / h - 0.5f;
        const float gy = y / h - 0.5f;
        const float gz = z / h;
        const int i1   = static_cast<int>(floorf(gx));
        const int j1   = static_cast<int>(floorf(gy));
        const int k1   = static_cast<int>(floorf(gz));
        const float tx = gx - static_cast<float>(i1);
        const float ty = gy - static_cast<float>(j1);
        const float tz = gz - static_cast<float>(k1);
        float z_samples[4];
        for (int dz = 0; dz < 4; ++dz) {
            float y_samples[4];
            for (int dy = 0; dy < 4; ++dy) {
                const int yy   = j1 + dy - 1;
                const int zz   = k1 + dz - 1;
                const float p0 = load_velocity_z(field, i1 - 1, yy, zz, nx, ny, nz, boundary);
                const float p1 = load_velocity_z(field, i1, yy, zz, nx, ny, nz, boundary);
                const float p2 = load_velocity_z(field, i1 + 1, yy, zz, nx, ny, nz, boundary);
                const float p3 = load_velocity_z(field, i1 + 2, yy, zz, nx, ny, nz, boundary);
                y_samples[dy]  = monotonic_cubic_1d(p0, p1, p2, p3, tx);
            }
            z_samples[dz] = monotonic_cubic_1d(y_samples[0], y_samples[1], y_samples[2], y_samples[3], ty);
        }
        return monotonic_cubic_1d(z_samples[0], z_samples[1], z_samples[2], z_samples[3], tz);
    }

    __device__ float3 sample_velocity(const float* velocity_x, const float* velocity_y, const float* velocity_z, const float x, const float y, const float z, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        return make_float3(sample_velocity_x(velocity_x, x, y, z, nx, ny, nz, h, boundary), sample_velocity_y(velocity_y, x, y, z, nx, ny, nz, h, boundary), sample_velocity_z(velocity_z, x, y, z, nx, ny, nz, h, boundary));
    }

    __device__ float3 trace_particle_rk2(const float3 start, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const float dt, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
        const auto [v0_x, v0_y, v0_z]    = sample_velocity(velocity_x, velocity_y, velocity_z, start.x, start.y, start.z, nx, ny, nz, h, boundary);
        const auto [mid_x, mid_y, mid_z] = make_float3(start.x - 0.5f * dt * v0_x, start.y - 0.5f * dt * v0_y, start.z - 0.5f * dt * v0_z);
        const auto [v1_x, v1_y, v1_z]    = sample_velocity(velocity_x, velocity_y, velocity_z, mid_x, mid_y, mid_z, nx, ny, nz, h, boundary);
        float3 traced                    = make_float3(start.x - dt * v1_x, start.y - dt * v1_y, start.z - dt * v1_z);
        float end_x                      = traced.x;
        float end_y                      = traced.y;
        float end_z                      = traced.z;
        if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
            const float extent_x = static_cast<float>(nx) * h;
            end_x                = fmodf(end_x, extent_x);
            if (end_x < 0.0f) end_x += extent_x;
        }
        if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
            const float extent_y = static_cast<float>(ny) * h;
            end_y                = fmodf(end_y, extent_y);
            if (end_y < 0.0f) end_y += extent_y;
        }
        if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
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
            if (boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nx > 0) {
                const float extent_x = static_cast<float>(nx) * h;
                test_x               = fmodf(test_x, extent_x);
                if (test_x < 0.0f) test_x += extent_x;
            }
            if (boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && ny > 0) {
                const float extent_y = static_cast<float>(ny) * h;
                test_y               = fmodf(test_y, extent_y);
                if (test_y < 0.0f) test_y += extent_y;
            }
            if (boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && nz > 0) {
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
            if (test_hits_solid)
                hi = mid_t;
            else
                lo = mid_t;
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

    __global__ void compute_vorticity_kernel(float* omega_x, float* omega_y, float* omega_z, float* omega_magnitude, const float* cell_x, const float* cell_y, const float* cell_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const SmokeSimulationFlowBoundaryConfig boundary) {
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

        const float dvz_dy = 0.5f * (load_center_velocity_component(cell_z, 2, x, y + 1, z, nx, ny, nz, boundary) - load_center_velocity_component(cell_z, 2, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float dvy_dz = 0.5f * (load_center_velocity_component(cell_y, 1, x, y, z + 1, nx, ny, nz, boundary) - load_center_velocity_component(cell_y, 1, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float dvx_dz = 0.5f * (load_center_velocity_component(cell_x, 0, x, y, z + 1, nx, ny, nz, boundary) - load_center_velocity_component(cell_x, 0, x, y, z - 1, nx, ny, nz, boundary)) / h;
        const float dvz_dx = 0.5f * (load_center_velocity_component(cell_z, 2, x + 1, y, z, nx, ny, nz, boundary) - load_center_velocity_component(cell_z, 2, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float dvy_dx = 0.5f * (load_center_velocity_component(cell_y, 1, x + 1, y, z, nx, ny, nz, boundary) - load_center_velocity_component(cell_y, 1, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float dvx_dy = 0.5f * (load_center_velocity_component(cell_x, 0, x, y + 1, z, nx, ny, nz, boundary) - load_center_velocity_component(cell_x, 0, x, y - 1, z, nx, ny, nz, boundary)) / h;

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

    __global__ void add_buoyancy_kernel(float* force_y, const float* density, const float* temperature, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float ambient_temperature, const float density_factor, const float temperature_factor, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;
        const auto index = index_3d(x, y, z, nx, ny);
        force_y[index] += -density_factor * density[index] + temperature_factor * (temperature[index] - ambient_temperature);
    }

    __global__ void add_confinement_kernel(float* force_x, float* force_y, float* force_z, const float* omega_x, const float* omega_y, const float* omega_z, const float* omega_magnitude, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float epsilon, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, boundary)) return;

        const float grad_x   = 0.5f * (load_flow_cell(omega_magnitude, x + 1, y, z, nx, ny, nz, boundary) - load_flow_cell(omega_magnitude, x - 1, y, z, nx, ny, nz, boundary)) / h;
        const float grad_y   = 0.5f * (load_flow_cell(omega_magnitude, x, y + 1, z, nx, ny, nz, boundary) - load_flow_cell(omega_magnitude, x, y - 1, z, nx, ny, nz, boundary)) / h;
        const float grad_z   = 0.5f * (load_flow_cell(omega_magnitude, x, y, z + 1, nx, ny, nz, boundary) - load_flow_cell(omega_magnitude, x, y, z - 1, nx, ny, nz, boundary)) / h;
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

    __device__ float solid_velocity_value(const float* solid_velocity, const uint8_t* occupancy, int x, int y, int z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        if (solid_velocity == nullptr || occupancy == nullptr) return 0.0f;
        if (!resolve_cell_coordinates(x, y, z, nx, ny, nz, boundary)) return 0.0f;
        if (occupancy[index_3d(x, y, z, nx, ny)] == 0) return 0.0f;
        return solid_velocity[index_3d(x, y, z, nx, ny)];
    }

    __global__ void enforce_velocity_x_boundaries_kernel(float* velocity_x, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[index_velocity_x(i, j, k, nx, ny)];
        if (i == 0) {
            if (const auto domain_face = boundary.x_minus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nx > 0)
                    face = velocity_x[index_velocity_x(1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }
        if (i == nx) {
            if (const auto domain_face = boundary.x_plus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nx > 0)
                    face = velocity_x[index_velocity_x(nx - 1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }
        if (occupancy == nullptr) return;

        int left_x                = i - 1;
        int left_y                = j;
        int left_z                = k;
        int right_x               = i;
        int right_y               = j;
        int right_z               = k;
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

    __global__ void enforce_velocity_y_boundaries_kernel(float* velocity_y, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[index_velocity_y(i, j, k, nx, ny)];
        if (j == 0) {
            if (const auto domain_face = boundary.y_minus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && ny > 0)
                    face = velocity_y[index_velocity_y(i, 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }
        if (j == ny) {
            if (const auto domain_face = boundary.y_plus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && ny > 0)
                    face = velocity_y[index_velocity_y(i, ny - 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }
        if (occupancy == nullptr) return;

        int down_x               = i;
        int down_y               = j - 1;
        int down_z               = k;
        int up_x                 = i;
        int up_y                 = j;
        int up_z                 = k;
        const bool has_down      = resolve_cell_coordinates(down_x, down_y, down_z, nx, ny, nz, boundary);
        const bool has_up        = resolve_cell_coordinates(up_x, up_y, up_z, nx, ny, nz, boundary);
        const bool down_occupied = has_down && occupancy[index_3d(down_x, down_y, down_z, nx, ny)] != 0;
        const bool up_occupied   = has_up && occupancy[index_3d(up_x, up_y, up_z, nx, ny)] != 0;
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

    __global__ void enforce_velocity_z_boundaries_kernel(float* velocity_z, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[index_velocity_z(i, j, k, nx, ny)];
        if (k == 0) {
            if (const auto domain_face = boundary.z_minus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nz > 0)
                    face = velocity_z[index_velocity_z(i, j, 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }
        if (k == nz) {
            if (const auto domain_face = boundary.z_plus; domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nz > 0)
                    face = velocity_z[index_velocity_z(i, j, nz - 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }
        if (occupancy == nullptr) return;

        int back_x                = i;
        int back_y                = j;
        int back_z                = k - 1;
        int front_x               = i;
        int front_y               = j;
        int front_z               = k;
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

    __global__ void advect_velocity_x_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t advection_mode, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;
        const float3 start                             = make_float3(static_cast<float>(i) * h, (static_cast<float>(j) + 0.5f) * h, (static_cast<float>(k) + 0.5f) * h);
        const auto [p_x, p_y, p_z]                     = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_x(i, j, k, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_velocity_x_cubic(source, p_x, p_y, p_z, nx, ny, nz, h, boundary) : sample_velocity_x(source, p_x, p_y, p_z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_velocity_y_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t advection_mode, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;
        const float3 start                             = make_float3((static_cast<float>(i) + 0.5f) * h, static_cast<float>(j) * h, (static_cast<float>(k) + 0.5f) * h);
        const auto [p_x, p_y, p_z]                     = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_y(i, j, k, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_velocity_y_cubic(source, p_x, p_y, p_z, nx, ny, nz, h, boundary) : sample_velocity_y(source, p_x, p_y, p_z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_velocity_z_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t advection_mode, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;
        const float3 start                             = make_float3((static_cast<float>(i) + 0.5f) * h, (static_cast<float>(j) + 0.5f) * h, static_cast<float>(k) * h);
        const auto [p_x, p_y, p_z]                     = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, boundary);
        destination[index_velocity_z(i, j, k, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_velocity_z_cubic(source, p_x, p_y, p_z, nx, ny, nz, h, boundary) : sample_velocity_z(source, p_x, p_y, p_z, nx, ny, nz, h, boundary);
    }

    __global__ void advect_scalar_kernel(float* destination, const float* source, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int nx, const int ny, const int nz, const float h, const float dt, const uint32_t advection_mode, const SmokeSimulationScalarBoundaryConfig scalar_boundary, const SmokeSimulationFlowBoundaryConfig flow_boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        if (load_occupancy(occupancy, x, y, z, nx, ny, nz, flow_boundary)) {
            destination[index_3d(x, y, z, nx, ny)] = 0.0f;
            return;
        }
        const float3 start                     = make_float3((static_cast<float>(x) + 0.5f) * h, (static_cast<float>(y) + 0.5f) * h, (static_cast<float>(z) + 0.5f) * h);
        const auto [p_x, p_y, p_z]             = trace_particle_rk2(start, velocity_x, velocity_y, velocity_z, occupancy, dt, nx, ny, nz, h, flow_boundary);
        destination[index_3d(x, y, z, nx, ny)] = advection_mode == SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC ? sample_scalar_cubic(source, p_x, p_y, p_z, nx, ny, nz, h, scalar_boundary) : sample_scalar_linear(source, p_x, p_y, p_z, nx, ny, nz, h, scalar_boundary);
    }

    __global__ void apply_solid_temperature_kernel(float* temperature, const uint8_t* occupancy, const float* solid_temperature, const int nx, const int ny, const int nz, const float ambient_temperature) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        const auto count = static_cast<std::uint64_t>(nx) * static_cast<std::uint64_t>(ny) * static_cast<std::uint64_t>(nz);
        if (index >= count) return;
        if (occupancy == nullptr || occupancy[index] == 0) return;
        temperature[index] = solid_temperature != nullptr ? solid_temperature[index] : ambient_temperature;
    }

    __global__ void boundary_fill_density_kernel(float* destination, const float* source, const uint8_t* occupancy, const int nx, const int ny, const int nz, const SmokeSimulationScalarBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;

        const auto index = index_3d(x, y, z, nx, ny);
        if (occupancy == nullptr || occupancy[index] == 0) {
            destination[index] = source[index];
            return;
        }

        int max_radius = nx;
        if (ny > max_radius) max_radius = ny;
        if (nz > max_radius) max_radius = nz;
        for (int radius = 1; radius <= max_radius; ++radius) {
            bool found         = false;
            float best_value   = 0.0f;
            int best_distance2 = 0;
            for (int dz = -radius; dz <= radius; ++dz) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int shell_radius = abs(dx);
                        if (abs(dy) > shell_radius) shell_radius = abs(dy);
                        if (abs(dz) > shell_radius) shell_radius = abs(dz);
                        if (shell_radius != radius) continue;
                        int next_x = x + dx;
                        int next_y = y + dy;
                        int next_z = z + dz;
                        if (!resolve_scalar_cell_coordinates(next_x, next_y, next_z, nx, ny, nz, boundary)) continue;
                        const auto neighbor_index = index_3d(next_x, next_y, next_z, nx, ny);
                        if (occupancy[neighbor_index] != 0) continue;
                        const int distance2 = dx * dx + dy * dy + dz * dz;
                        if (!found || distance2 < best_distance2) {
                            found          = true;
                            best_distance2 = distance2;
                            best_value     = source[neighbor_index];
                        }
                    }
                }
            }
            if (found) {
                destination[index] = best_value;
                return;
            }
        }
        destination[index] = 0.0f;
    }

    __global__ void initialize_pressure_anchor_kernel(int* pressure_anchor, const int cell_count) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *pressure_anchor = cell_count;
    }

    __global__ void find_pressure_anchor_kernel(int* pressure_anchor, const uint8_t* occupancy, const std::uint64_t count) {
        const auto index = static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) + static_cast<std::uint64_t>(threadIdx.x);
        if (index >= count) return;
        if (occupancy != nullptr && occupancy[index] != 0) return;
        atomicMin(pressure_anchor, static_cast<int>(index));
    }

    __global__ void compute_pressure_rhs_kernel(float* rhs, const float* velocity_x, const float* velocity_y, const float* velocity_z, const uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (x >= nx || y >= ny || z >= nz) return;
        const auto index = index_3d(x, y, z, nx, ny);
        const int anchor = *pressure_anchor;
        if (static_cast<int>(index) == anchor) {
            rhs[index] = 0.0f;
            return;
        }
        if (occupancy != nullptr && occupancy[index] != 0) {
            rhs[index] = 0.0f;
            return;
        }
        const float divergence = (velocity_x[index_velocity_x(x + 1, y, z, nx, ny)] - velocity_x[index_velocity_x(x, y, z, nx, ny)] + velocity_y[index_velocity_y(x, y + 1, z, nx, ny)] - velocity_y[index_velocity_y(x, y, z, nx, ny)] + velocity_z[index_velocity_z(x, y, z + 1, nx, ny)] - velocity_z[index_velocity_z(x, y, z, nx, ny)]) / h;
        float boundary_sum     = 0.0f;
        if (x == 0 && boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.x_minus.pressure;
        if (x == nx - 1 && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.x_plus.pressure;
        if (y == 0 && boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.y_minus.pressure;
        if (y == ny - 1 && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.y_plus.pressure;
        if (z == 0 && boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.z_minus.pressure;
        if (z == nz - 1 && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) boundary_sum += boundary.z_plus.pressure;
        rhs[index]             = -(h * h / dt) * divergence + boundary_sum;
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
        divergence[index] = (velocity_x[index_velocity_x(x + 1, y, z, nx, ny)] - velocity_x[index_velocity_x(x, y, z, nx, ny)] + velocity_y[index_velocity_y(x, y + 1, z, nx, ny)] - velocity_y[index_velocity_y(x, y, z, nx, ny)] + velocity_z[index_velocity_z(x, y, z + 1, nx, ny)] - velocity_z[index_velocity_z(x, y, z, nx, ny)]) / h;
    }

    __global__ void build_pressure_matrix_values_kernel(float* values, float* factor_values, const int* row_offsets, const int* column_indices, const int* factor_row_offsets, const int* factor_column_indices, const uint8_t* occupancy, const int* pressure_anchor, const int nx, const int ny, const int nz, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (row >= nx * ny * nz) return;

        const int anchor      = *pressure_anchor;
        const int x           = row % nx;
        const int yz          = row / nx;
        const int y           = yz % ny;
        const int z           = yz / ny;
        const bool occupied   = occupancy != nullptr && occupancy[static_cast<std::uint64_t>(row)] != 0;
        const bool special    = occupied || row == anchor;
        const bool periodic_x = boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_y = boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_z = boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;

        std::array<int, 6> active_neighbors{};
        int active_neighbor_count = 0;
        float diagonal            = 0.0f;

        auto accumulate_neighbor = [&](int next_x, int next_y, int next_z, const SmokeSimulationFlowBoundaryFaceDesc minus_face, const SmokeSimulationFlowBoundaryFaceDesc plus_face, const bool periodic_axis) {
            if (next_x < 0 || next_x >= nx || next_y < 0 || next_y >= ny || next_z < 0 || next_z >= nz) {
                if (periodic_axis) {
                    if (next_x < 0 || next_x >= nx) next_x = wrap_index(next_x, nx);
                    if (next_y < 0 || next_y >= ny) next_y = wrap_index(next_y, ny);
                    if (next_z < 0 || next_z >= nz) next_z = wrap_index(next_z, nz);
                } else {
                    const auto face = next_x < 0 || next_y < 0 || next_z < 0 ? minus_face : plus_face;
                    if (face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW) diagonal += 1.0f;
                    return;
                }
            }
            const int neighbor = static_cast<int>(index_3d(next_x, next_y, next_z, nx, ny));
            if (occupancy != nullptr && occupancy[static_cast<std::uint64_t>(neighbor)] != 0) return;
            diagonal += 1.0f;
            if (neighbor == anchor) return;
            for (int index = 0; index < active_neighbor_count; ++index) {
                if (active_neighbors[index] == neighbor) return;
            }
            active_neighbors[active_neighbor_count] = neighbor;
            ++active_neighbor_count;
        };

        if (!special) {
            accumulate_neighbor(x - 1, y, z, boundary.x_minus, boundary.x_plus, periodic_x);
            accumulate_neighbor(x + 1, y, z, boundary.x_minus, boundary.x_plus, periodic_x);
            accumulate_neighbor(x, y - 1, z, boundary.y_minus, boundary.y_plus, periodic_y);
            accumulate_neighbor(x, y + 1, z, boundary.y_minus, boundary.y_plus, periodic_y);
            accumulate_neighbor(x, y, z - 1, boundary.z_minus, boundary.z_plus, periodic_z);
            accumulate_neighbor(x, y, z + 1, boundary.z_minus, boundary.z_plus, periodic_z);
            if (diagonal <= 0.0f) diagonal = 1.0f;
        }

        for (int entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
            const int column = column_indices[entry];
            float value      = 0.0f;
            if (special) {
                value = column == row ? 1.0f : 0.0f;
            } else if (column == row) {
                value = diagonal;
            } else {
                for (int index = 0; index < active_neighbor_count; ++index) {
                    if (active_neighbors[index] == column) {
                        value = -1.0f;
                        break;
                    }
                }
            }
            values[entry] = value;
        }

        for (int entry = factor_row_offsets[row]; entry < factor_row_offsets[row + 1]; ++entry) {
            const int column = factor_column_indices[entry];
            float value      = 0.0f;
            if (special) {
                value = column == row ? 1.0f : 0.0f;
            } else if (column == row) {
                value = diagonal;
            } else {
                for (int index = 0; index < active_neighbor_count; ++index) {
                    if (active_neighbors[index] == column) {
                        value = -1.0f;
                        break;
                    }
                }
            }
            factor_values[entry] = value;
        }
    }

    __global__ void compute_ratio_kernel(float* destination, const float* numerator, const float* denominator) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        const float value = fabsf(*denominator) > 1.0e-20f ? *numerator / *denominator : 0.0f;
        *destination      = value;
    }

    __global__ void negate_scalar_kernel(float* destination, const float* source) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *destination = -*source;
    }

    __global__ void project_velocity_x_kernel(float* velocity_x, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_x, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i > nx || j >= ny || k >= nz) return;

        auto& face = velocity_x[index_velocity_x(i, j, k, nx, ny)];
        if (i == 0) {
            const auto domain_face = boundary.x_minus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nx > 0)
                    face = velocity_x[index_velocity_x(1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }
        if (i == nx) {
            const auto domain_face = boundary.x_plus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nx > 0)
                    face = velocity_x[index_velocity_x(nx - 1, j, k, nx, ny)];
                else
                    face = domain_face.velocity_x;
                return;
            }
        }

        int left_x                = i - 1;
        int left_y                = j;
        int left_z                = k;
        int right_x               = i;
        int right_y               = j;
        int right_z               = k;
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

    __global__ void project_velocity_y_kernel(float* velocity_y, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_y, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j > ny || k >= nz) return;

        auto& face = velocity_y[index_velocity_y(i, j, k, nx, ny)];
        if (j == 0) {
            const auto domain_face = boundary.y_minus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && ny > 0)
                    face = velocity_y[index_velocity_y(i, 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }
        if (j == ny) {
            const auto domain_face = boundary.y_plus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && ny > 0)
                    face = velocity_y[index_velocity_y(i, ny - 1, k, nx, ny)];
                else
                    face = domain_face.velocity_y;
                return;
            }
        }

        int down_x               = i;
        int down_y               = j - 1;
        int down_z               = k;
        int up_x                 = i;
        int up_y                 = j;
        int up_z                 = k;
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

    __global__ void project_velocity_z_kernel(float* velocity_z, const float* pressure, const uint8_t* occupancy, const float* solid_velocity_z, const int nx, const int ny, const int nz, const float h, const float dt, const SmokeSimulationFlowBoundaryConfig boundary) {
        const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
        const int k = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
        if (i >= nx || j >= ny || k > nz) return;

        auto& face = velocity_z[index_velocity_z(i, j, k, nx, ny)];
        if (k == 0) {
            const auto domain_face = boundary.z_minus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nz > 0)
                    face = velocity_z[index_velocity_z(i, j, 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }
        if (k == nz) {
            const auto domain_face = boundary.z_plus;
            if (domain_face.type != SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) {
                if (domain_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW && nz > 0)
                    face = velocity_z[index_velocity_z(i, j, nz - 1, nx, ny)];
                else
                    face = domain_face.velocity_z;
                return;
            }
        }

        int back_x                = i;
        int back_y                = j;
        int back_z                = k - 1;
        int front_x               = i;
        int front_y               = j;
        int front_z               = k;
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

    void check_cuda(const cudaError_t status, const char* what) {
        if (status == cudaSuccess) return;
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }


    void destroy_context_resources(ContextStorage& context) {
        if (context.step_graph.exec != nullptr) cudaGraphExecDestroy(context.step_graph.exec);
        if (context.step_graph.graph != nullptr) cudaGraphDestroy(context.step_graph.graph);
        context.step_graph.exec  = nullptr;
        context.step_graph.graph = nullptr;
        if (context.pressure_solver.matrix != nullptr) cusparseDestroySpMat(context.pressure_solver.matrix);
        if (context.pressure_solver.factor != nullptr) cusparseDestroySpMat(context.pressure_solver.factor);
        if (context.pressure_solver.vec_r != nullptr) cusparseDestroyDnVec(context.pressure_solver.vec_r);
        if (context.pressure_solver.vec_p != nullptr) cusparseDestroyDnVec(context.pressure_solver.vec_p);
        if (context.pressure_solver.vec_ap != nullptr) cusparseDestroyDnVec(context.pressure_solver.vec_ap);
        if (context.pressure_solver.vec_y != nullptr) cusparseDestroyDnVec(context.pressure_solver.vec_y);
        if (context.pressure_solver.vec_z != nullptr) cusparseDestroyDnVec(context.pressure_solver.vec_z);
        if (context.pressure_solver.lower_solve != nullptr) cusparseSpSV_destroyDescr(context.pressure_solver.lower_solve);
        if (context.pressure_solver.upper_solve != nullptr) cusparseSpSV_destroyDescr(context.pressure_solver.upper_solve);
        if (context.pressure_solver.factor_info != nullptr) cusparseDestroyCsric02Info(context.pressure_solver.factor_info);
        if (context.pressure_solver.factor_descr != nullptr) cusparseDestroyMatDescr(context.pressure_solver.factor_descr);
        if (context.pressure_solver.factor_buffer != nullptr) cudaFree(context.pressure_solver.factor_buffer);
        if (context.pressure_solver.spmv_buffer != nullptr) cudaFree(context.pressure_solver.spmv_buffer);
        if (context.pressure_solver.lower_solve_buffer != nullptr) cudaFree(context.pressure_solver.lower_solve_buffer);
        if (context.pressure_solver.upper_solve_buffer != nullptr) cudaFree(context.pressure_solver.upper_solve_buffer);
        if (context.pressure_solver.cublas != nullptr) cublasDestroy(context.pressure_solver.cublas);
        if (context.pressure_solver.cusparse != nullptr) cusparseDestroy(context.pressure_solver.cusparse);
        context.pressure_solver = ContextStorage::PressureSolverStorage{};
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
        if (context.device.flow.pressure_anchor != nullptr) cudaFree(context.device.flow.pressure_anchor);
        if (context.device.flow.pressure_row_offsets != nullptr) cudaFree(context.device.flow.pressure_row_offsets);
        if (context.device.flow.pressure_column_indices != nullptr) cudaFree(context.device.flow.pressure_column_indices);
        if (context.device.flow.pressure_values != nullptr) cudaFree(context.device.flow.pressure_values);
        if (context.device.flow.pressure_factor_row_offsets != nullptr) cudaFree(context.device.flow.pressure_factor_row_offsets);
        if (context.device.flow.pressure_factor_column_indices != nullptr) cudaFree(context.device.flow.pressure_factor_column_indices);
        if (context.device.flow.pressure_factor_values != nullptr) cudaFree(context.device.flow.pressure_factor_values);
        if (context.device.flow.pcg_r != nullptr) cudaFree(context.device.flow.pcg_r);
        if (context.device.flow.pcg_p != nullptr) cudaFree(context.device.flow.pcg_p);
        if (context.device.flow.pcg_ap != nullptr) cudaFree(context.device.flow.pcg_ap);
        if (context.device.flow.pcg_z != nullptr) cudaFree(context.device.flow.pcg_z);
        if (context.device.flow.pcg_y != nullptr) cudaFree(context.device.flow.pcg_y);
        if (context.device.flow.pressure_dot_rz != nullptr) cudaFree(context.device.flow.pressure_dot_rz);
        if (context.device.flow.pressure_dot_pap != nullptr) cudaFree(context.device.flow.pressure_dot_pap);
        if (context.device.flow.pressure_dot_rr != nullptr) cudaFree(context.device.flow.pressure_dot_rr);
        if (context.device.flow.pressure_alpha != nullptr) cudaFree(context.device.flow.pressure_alpha);
        if (context.device.flow.pressure_negative_alpha != nullptr) cudaFree(context.device.flow.pressure_negative_alpha);
        if (context.device.flow.pressure_beta != nullptr) cudaFree(context.device.flow.pressure_beta);
        if (context.device.flow.pressure_one != nullptr) cudaFree(context.device.flow.pressure_one);
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
        context.device.occupancy_float   = nullptr;
        context.device.occupancy         = nullptr;
        context.device.solid_temperature = nullptr;
    }

    void initialize_pressure_system(ContextStorage& context) {
        auto& flow       = context.device.flow;
        const int cells  = static_cast<int>(context.cell_count);
        const bool periodic_x = context.config.flow_boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context.config.flow_boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_y = context.config.flow_boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context.config.flow_boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_z = context.config.flow_boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context.config.flow_boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        std::vector<int> host_row_offsets(static_cast<std::size_t>(cells) + 1u, 0);
        std::vector<int> host_factor_row_offsets(static_cast<std::size_t>(cells) + 1u, 0);
        std::vector<int> host_column_indices{};
        std::vector<int> host_factor_column_indices{};
        host_column_indices.reserve(static_cast<std::size_t>(cells) * 7u);
        host_factor_column_indices.reserve(static_cast<std::size_t>(cells) * 4u);

        for (int row = 0; row < cells; ++row) {
            host_row_offsets[static_cast<std::size_t>(row)]        = static_cast<int>(host_column_indices.size());
            host_factor_row_offsets[static_cast<std::size_t>(row)] = static_cast<int>(host_factor_column_indices.size());

            const int x = row % context.config.nx;
            const int yz = row / context.config.nx;
            const int y = yz % context.config.ny;
            const int z = yz / context.config.ny;
            std::array<int, 7> row_columns{};
            int row_entry_count = 0;

            auto add_column = [&](const int column) {
                for (int entry = 0; entry < row_entry_count; ++entry) {
                    if (row_columns[entry] == column) return;
                }
                row_columns[row_entry_count] = column;
                ++row_entry_count;
            };
            auto accumulate_neighbor = [&](int next_x, int next_y, int next_z, const bool periodic_axis) {
                if (next_x < 0 || next_x >= context.config.nx || next_y < 0 || next_y >= context.config.ny || next_z < 0 || next_z >= context.config.nz) {
                    if (periodic_axis) {
                        if (next_x < 0 || next_x >= context.config.nx) next_x = wrap_index(next_x, context.config.nx);
                        if (next_y < 0 || next_y >= context.config.ny) next_y = wrap_index(next_y, context.config.ny);
                        if (next_z < 0 || next_z >= context.config.nz) next_z = wrap_index(next_z, context.config.nz);
                    } else {
                        return;
                    }
                }
                add_column(static_cast<int>(index_3d(next_x, next_y, next_z, context.config.nx, context.config.ny)));
            };

            accumulate_neighbor(x - 1, y, z, periodic_x);
            accumulate_neighbor(x + 1, y, z, periodic_x);
            accumulate_neighbor(x, y - 1, z, periodic_y);
            accumulate_neighbor(x, y + 1, z, periodic_y);
            accumulate_neighbor(x, y, z - 1, periodic_z);
            accumulate_neighbor(x, y, z + 1, periodic_z);
            add_column(row);

            for (int left = 0; left < row_entry_count; ++left) {
                for (int right = left + 1; right < row_entry_count; ++right) {
                    if (row_columns[right] < row_columns[left]) {
                        const int swapped_column = row_columns[left];
                        row_columns[left]        = row_columns[right];
                        row_columns[right]       = swapped_column;
                    }
                }
            }

            for (int entry = 0; entry < row_entry_count; ++entry) {
                host_column_indices.push_back(row_columns[entry]);
                if (row_columns[entry] <= row) host_factor_column_indices.push_back(row_columns[entry]);
            }
        }

        host_row_offsets[static_cast<std::size_t>(cells)]        = static_cast<int>(host_column_indices.size());
        host_factor_row_offsets[static_cast<std::size_t>(cells)] = static_cast<int>(host_factor_column_indices.size());
        flow.pressure_nnz                                         = static_cast<int>(host_column_indices.size());
        flow.pressure_factor_nnz                                  = static_cast<int>(host_factor_column_indices.size());

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_anchor), sizeof(int)), "cudaMalloc pressure_anchor");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_row_offsets), static_cast<std::size_t>(cells + 1) * sizeof(int)), "cudaMalloc pressure_row_offsets");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_column_indices), static_cast<std::size_t>(flow.pressure_nnz) * sizeof(int)), "cudaMalloc pressure_column_indices");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_values), static_cast<std::size_t>(flow.pressure_nnz) * sizeof(float)), "cudaMalloc pressure_values");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_factor_row_offsets), static_cast<std::size_t>(cells + 1) * sizeof(int)), "cudaMalloc pressure_factor_row_offsets");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_factor_column_indices), static_cast<std::size_t>(flow.pressure_factor_nnz) * sizeof(int)), "cudaMalloc pressure_factor_column_indices");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_factor_values), static_cast<std::size_t>(flow.pressure_factor_nnz) * sizeof(float)), "cudaMalloc pressure_factor_values");
        check_cuda(cudaMemcpyAsync(flow.pressure_row_offsets, host_row_offsets.data(), static_cast<std::size_t>(cells + 1) * sizeof(int), cudaMemcpyHostToDevice, context.stream), "cudaMemcpyAsync pressure_row_offsets");
        check_cuda(cudaMemcpyAsync(flow.pressure_column_indices, host_column_indices.data(), static_cast<std::size_t>(flow.pressure_nnz) * sizeof(int), cudaMemcpyHostToDevice, context.stream), "cudaMemcpyAsync pressure_column_indices");
        check_cuda(cudaMemsetAsync(flow.pressure_values, 0, static_cast<std::size_t>(flow.pressure_nnz) * sizeof(float), context.stream), "cudaMemsetAsync pressure_values");
        check_cuda(cudaMemcpyAsync(flow.pressure_factor_row_offsets, host_factor_row_offsets.data(), static_cast<std::size_t>(cells + 1) * sizeof(int), cudaMemcpyHostToDevice, context.stream), "cudaMemcpyAsync pressure_factor_row_offsets");
        check_cuda(cudaMemcpyAsync(flow.pressure_factor_column_indices, host_factor_column_indices.data(), static_cast<std::size_t>(flow.pressure_factor_nnz) * sizeof(int), cudaMemcpyHostToDevice, context.stream), "cudaMemcpyAsync pressure_factor_column_indices");
        check_cuda(cudaMemsetAsync(flow.pressure_factor_values, 0, static_cast<std::size_t>(flow.pressure_factor_nnz) * sizeof(float), context.stream), "cudaMemsetAsync pressure_factor_values");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pcg_r), context.cell_bytes), "cudaMalloc pcg_r");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pcg_p), context.cell_bytes), "cudaMalloc pcg_p");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pcg_ap), context.cell_bytes), "cudaMalloc pcg_ap");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pcg_z), context.cell_bytes), "cudaMalloc pcg_z");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pcg_y), context.cell_bytes), "cudaMalloc pcg_y");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_dot_rz), sizeof(float)), "cudaMalloc pressure_dot_rz");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_dot_pap), sizeof(float)), "cudaMalloc pressure_dot_pap");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_dot_rr), sizeof(float)), "cudaMalloc pressure_dot_rr");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_alpha), sizeof(float)), "cudaMalloc pressure_alpha");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_negative_alpha), sizeof(float)), "cudaMalloc pressure_negative_alpha");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_beta), sizeof(float)), "cudaMalloc pressure_beta");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&flow.pressure_one), sizeof(float)), "cudaMalloc pressure_one");
        const float one = 1.0f;
        check_cuda(cudaMemcpyAsync(flow.pressure_one, &one, sizeof(float), cudaMemcpyHostToDevice, context.stream), "cudaMemcpyAsync pressure_one");
        check_cuda(cudaMemsetAsync(flow.pcg_r, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pcg_r");
        check_cuda(cudaMemsetAsync(flow.pcg_p, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pcg_p");
        check_cuda(cudaMemsetAsync(flow.pcg_ap, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pcg_ap");
        check_cuda(cudaMemsetAsync(flow.pcg_z, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pcg_z");
        check_cuda(cudaMemsetAsync(flow.pcg_y, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pcg_y");
        check_cuda(cudaStreamSynchronize(context.stream), "cudaStreamSynchronize pressure_system_upload");

        if (cusparseCreateMatDescr(&context.pressure_solver.factor_descr) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateMatDescr");
        cusparseSetMatType(context.pressure_solver.factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(context.pressure_solver.factor_descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatFillMode(context.pressure_solver.factor_descr, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(context.pressure_solver.factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
        if (cusparseCreateCsric02Info(&context.pressure_solver.factor_info) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateCsric02Info");
        int factor_buffer_size = 0;
        if (cusparseScsric02_bufferSize(context.pressure_solver.cusparse, cells, flow.pressure_factor_nnz, context.pressure_solver.factor_descr, flow.pressure_factor_values, flow.pressure_factor_row_offsets, flow.pressure_factor_column_indices, context.pressure_solver.factor_info, &factor_buffer_size) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseScsric02_bufferSize");
        context.pressure_solver.factor_buffer_size = static_cast<std::size_t>(factor_buffer_size);
        if (context.pressure_solver.factor_buffer_size > 0) check_cuda(cudaMalloc(&context.pressure_solver.factor_buffer, context.pressure_solver.factor_buffer_size), "cudaMalloc factor_buffer");
        if (cusparseScsric02_analysis(context.pressure_solver.cusparse, cells, flow.pressure_factor_nnz, context.pressure_solver.factor_descr, flow.pressure_factor_values, flow.pressure_factor_row_offsets, flow.pressure_factor_column_indices, context.pressure_solver.factor_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, context.pressure_solver.factor_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseScsric02_analysis");
        if (cusparseCreateCsr(&context.pressure_solver.matrix, cells, cells, flow.pressure_nnz, flow.pressure_row_offsets, flow.pressure_column_indices, flow.pressure_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateCsr matrix");
        if (cusparseCreateCsr(&context.pressure_solver.factor, cells, cells, flow.pressure_factor_nnz, flow.pressure_factor_row_offsets, flow.pressure_factor_column_indices, flow.pressure_factor_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateCsr factor");
        cusparseFillMode_t fill_mode = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
        if (cusparseSpMatSetAttribute(context.pressure_solver.factor, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(fill_mode)) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpMatSetAttribute fill_mode");
        if (cusparseSpMatSetAttribute(context.pressure_solver.factor, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type, sizeof(diag_type)) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpMatSetAttribute diag_type");
        if (cusparseCreateDnVec(&context.pressure_solver.vec_r, cells, flow.pcg_r, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateDnVec vec_r");
        if (cusparseCreateDnVec(&context.pressure_solver.vec_p, cells, flow.pcg_p, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateDnVec vec_p");
        if (cusparseCreateDnVec(&context.pressure_solver.vec_ap, cells, flow.pcg_ap, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateDnVec vec_ap");
        if (cusparseCreateDnVec(&context.pressure_solver.vec_y, cells, flow.pcg_y, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateDnVec vec_y");
        if (cusparseCreateDnVec(&context.pressure_solver.vec_z, cells, flow.pcg_z, CUDA_R_32F) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreateDnVec vec_z");
        const float spmv_alpha = 1.0f;
        const float spmv_beta  = 0.0f;
        if (cusparseSpMV_bufferSize(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, context.pressure_solver.matrix, context.pressure_solver.vec_p, &spmv_beta, context.pressure_solver.vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &context.pressure_solver.spmv_buffer_size) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpMV_bufferSize");
        if (context.pressure_solver.spmv_buffer_size > 0) check_cuda(cudaMalloc(&context.pressure_solver.spmv_buffer, context.pressure_solver.spmv_buffer_size), "cudaMalloc spmv_buffer");
        if (cusparseSpMV_preprocess(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, context.pressure_solver.matrix, context.pressure_solver.vec_p, &spmv_beta, context.pressure_solver.vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, context.pressure_solver.spmv_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpMV_preprocess");
        if (cusparseSpSV_createDescr(&context.pressure_solver.lower_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_createDescr lower");
        if (cusparseSpSV_createDescr(&context.pressure_solver.upper_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_createDescr upper");
        if (cusparseSpSV_bufferSize(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, context.pressure_solver.factor, context.pressure_solver.vec_r, context.pressure_solver.vec_y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.lower_solve, &context.pressure_solver.lower_solve_buffer_size) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_bufferSize lower");
        if (context.pressure_solver.lower_solve_buffer_size > 0) check_cuda(cudaMalloc(&context.pressure_solver.lower_solve_buffer, context.pressure_solver.lower_solve_buffer_size), "cudaMalloc lower_solve_buffer");
        if (cusparseSpSV_analysis(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, context.pressure_solver.factor, context.pressure_solver.vec_r, context.pressure_solver.vec_y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.lower_solve, context.pressure_solver.lower_solve_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_analysis lower");
        if (cusparseSpSV_bufferSize(context.pressure_solver.cusparse, CUSPARSE_OPERATION_TRANSPOSE, &spmv_alpha, context.pressure_solver.factor, context.pressure_solver.vec_y, context.pressure_solver.vec_z, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.upper_solve, &context.pressure_solver.upper_solve_buffer_size) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_bufferSize upper");
        if (context.pressure_solver.upper_solve_buffer_size > 0) check_cuda(cudaMalloc(&context.pressure_solver.upper_solve_buffer, context.pressure_solver.upper_solve_buffer_size), "cudaMalloc upper_solve_buffer");
        if (cusparseSpSV_analysis(context.pressure_solver.cusparse, CUSPARSE_OPERATION_TRANSPOSE, &spmv_alpha, context.pressure_solver.factor, context.pressure_solver.vec_y, context.pressure_solver.vec_z, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.upper_solve, context.pressure_solver.upper_solve_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_analysis upper");
    }

    void solve_pressure_pcg(ContextStorage& context) {
        auto& flow = context.device.flow;
        constexpr unsigned block_size = 256u;
        const unsigned linear_grid = static_cast<unsigned>((context.cell_count + block_size - 1u) / block_size);
        initialize_pressure_anchor_kernel<<<1, 1, 0, context.stream>>>(flow.pressure_anchor, static_cast<int>(context.cell_count));
        check_cuda(cudaGetLastError(), "initialize_pressure_anchor_kernel");
        find_pressure_anchor_kernel<<<linear_grid, block_size, 0, context.stream>>>(flow.pressure_anchor, context.device.occupancy, context.cell_count);
        check_cuda(cudaGetLastError(), "find_pressure_anchor_kernel");
        compute_pressure_rhs_kernel<<<context.cells, context.block, 0, context.stream>>>(flow.pressure_rhs, flow.velocity_x, flow.velocity_y, flow.velocity_z, context.device.occupancy, flow.pressure_anchor, context.config.nx, context.config.ny, context.config.nz, context.config.cell_size, context.config.dt, context.config.flow_boundary);
        check_cuda(cudaGetLastError(), "compute_pressure_rhs_kernel");
        build_pressure_matrix_values_kernel<<<linear_grid, block_size, 0, context.stream>>>(flow.pressure_values, flow.pressure_factor_values, flow.pressure_row_offsets, flow.pressure_column_indices, flow.pressure_factor_row_offsets, flow.pressure_factor_column_indices, context.device.occupancy, flow.pressure_anchor, context.config.nx, context.config.ny, context.config.nz, context.config.flow_boundary);
        check_cuda(cudaGetLastError(), "build_pressure_matrix_values_kernel");
        check_cuda(cudaMemsetAsync(flow.pressure, 0, context.cell_bytes, context.stream), "cudaMemsetAsync pressure");
        if (cusparseScsric02(context.pressure_solver.cusparse, static_cast<int>(context.cell_count), flow.pressure_factor_nnz, context.pressure_solver.factor_descr, flow.pressure_factor_values, flow.pressure_factor_row_offsets, flow.pressure_factor_column_indices, context.pressure_solver.factor_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, context.pressure_solver.factor_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseScsric02");
        if (cublasScopy(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pressure_rhs, 1, flow.pcg_r, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasScopy rhs");

        const float one  = 1.0f;
        const float zero = 0.0f;
        if (cusparseSpSV_solve(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, context.pressure_solver.factor, context.pressure_solver.vec_r, context.pressure_solver.vec_y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.lower_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_solve lower");
        if (cusparseSpSV_solve(context.pressure_solver.cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one, context.pressure_solver.factor, context.pressure_solver.vec_y, context.pressure_solver.vec_z, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.upper_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_solve upper");
        if (cublasScopy(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pcg_z, 1, flow.pcg_p, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasScopy pcg_p");
        if (cublasSdot(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pcg_r, 1, flow.pcg_z, 1, flow.pressure_dot_rz) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSdot pressure_dot_rz");

        for (int iteration = 0; iteration < context.config.pressure_iterations; ++iteration) {
            if (cusparseSpMV(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, context.pressure_solver.matrix, context.pressure_solver.vec_p, &zero, context.pressure_solver.vec_ap, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, context.pressure_solver.spmv_buffer) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpMV");
            if (cublasSdot(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pcg_p, 1, flow.pcg_ap, 1, flow.pressure_dot_pap) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSdot pressure_dot_pap");
            compute_ratio_kernel<<<1, 1, 0, context.stream>>>(flow.pressure_alpha, flow.pressure_dot_rz, flow.pressure_dot_pap);
            check_cuda(cudaGetLastError(), "compute_ratio_kernel alpha");
            if (cublasSaxpy(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pressure_alpha, flow.pcg_p, 1, flow.pressure, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSaxpy pressure");
            negate_scalar_kernel<<<1, 1, 0, context.stream>>>(flow.pressure_negative_alpha, flow.pressure_alpha);
            check_cuda(cudaGetLastError(), "negate_scalar_kernel");
            if (cublasSaxpy(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pressure_negative_alpha, flow.pcg_ap, 1, flow.pcg_r, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSaxpy pcg_r");
            if (cusparseSpSV_solve(context.pressure_solver.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, context.pressure_solver.factor, context.pressure_solver.vec_r, context.pressure_solver.vec_y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.lower_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_solve lower iterate");
            if (cusparseSpSV_solve(context.pressure_solver.cusparse, CUSPARSE_OPERATION_TRANSPOSE, &one, context.pressure_solver.factor, context.pressure_solver.vec_y, context.pressure_solver.vec_z, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, context.pressure_solver.upper_solve) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSpSV_solve upper iterate");
            if (cublasSdot(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pcg_r, 1, flow.pcg_z, 1, flow.pressure_dot_rr) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSdot rho_new");
            compute_ratio_kernel<<<1, 1, 0, context.stream>>>(flow.pressure_beta, flow.pressure_dot_rr, flow.pressure_dot_rz);
            check_cuda(cudaGetLastError(), "compute_ratio_kernel beta");
            if (cublasSscal(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pressure_beta, flow.pcg_p, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSscal pcg_p");
            if (cublasSaxpy(context.pressure_solver.cublas, static_cast<int>(context.cell_count), flow.pressure_one, flow.pcg_z, 1, flow.pcg_p, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSaxpy pcg_p");
            if (cublasScopy(context.pressure_solver.cublas, 1, flow.pressure_dot_rr, 1, flow.pressure_dot_rz, 1) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasScopy rho");
        }
    }


} // namespace smoke_simulation

struct SmokeSimulationContext_t : smoke_simulation::ContextStorage {};

extern "C" {

SmokeSimulationResult smoke_simulation_create_context_cuda(const SmokeSimulationContextCreateDesc* desc, SmokeSimulationContext* out_context) {
    nvtx3::scoped_range range("smoke.create_context");
    if (desc == nullptr || out_context == nullptr) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if (desc->config.nx <= 0 || desc->config.ny <= 0 || desc->config.nz <= 0 || desc->config.cell_size <= 0.0f || desc->config.dt <= 0.0f || desc->config.pressure_iterations <= 0) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    *out_context = nullptr;

    std::unique_ptr<SmokeSimulationContext_t> context{new (std::nothrow) SmokeSimulationContext_t{}};
    if (!context) return SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY;

    try {
        context->config        = desc->config;
        context->stream        = static_cast<cudaStream_t>(desc->stream);
        auto valid_flow_face   = [](const SmokeSimulationFlowBoundaryFaceDesc face) { return face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_NO_SLIP_WALL || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW || face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC; };
        auto valid_scalar_face = [](const SmokeSimulationScalarBoundaryFaceDesc face) { return face.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE || face.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_ZERO_FLUX || face.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC; };
        auto valid_flow_axis   = [&](const SmokeSimulationFlowBoundaryFaceDesc minus_face, const SmokeSimulationFlowBoundaryFaceDesc plus_face) {
            if (!valid_flow_face(minus_face) || !valid_flow_face(plus_face)) return false;
            return (minus_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC) == (plus_face.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC);
        };
        auto valid_scalar_axis = [&](const SmokeSimulationScalarBoundaryFaceDesc minus_face, const SmokeSimulationScalarBoundaryFaceDesc plus_face) {
            if (!valid_scalar_face(minus_face) || !valid_scalar_face(plus_face)) return false;
            return (minus_face.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC) == (plus_face.type == SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC);
        };
        if (!valid_flow_axis(context->config.flow_boundary.x_minus, context->config.flow_boundary.x_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_flow_axis(context->config.flow_boundary.y_minus, context->config.flow_boundary.y_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_flow_axis(context->config.flow_boundary.z_minus, context->config.flow_boundary.z_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.density_boundary.x_minus, context->config.density_boundary.x_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.density_boundary.y_minus, context->config.density_boundary.y_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.density_boundary.z_minus, context->config.density_boundary.z_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.temperature_boundary.x_minus, context->config.temperature_boundary.x_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.temperature_boundary.y_minus, context->config.temperature_boundary.y_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (!valid_scalar_axis(context->config.temperature_boundary.z_minus, context->config.temperature_boundary.z_plus)) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        context->cell_count       = static_cast<std::uint64_t>(context->config.nx) * static_cast<std::uint64_t>(context->config.ny) * static_cast<std::uint64_t>(context->config.nz);
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
        if (cublasCreate(&context->pressure_solver.cublas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        if (cublasSetStream(context->pressure_solver.cublas, context->stream) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSetStream");
        if (cublasSetPointerMode(context->pressure_solver.cublas, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasSetPointerMode");
        if (cusparseCreate(&context->pressure_solver.cusparse) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseCreate");
        if (cusparseSetStream(context->pressure_solver.cusparse, context->stream) != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("cusparseSetStream");
        auto choose_block = [&]() {
            int min_grid_size = 0;
            int block_size    = 0;
            if (cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, smoke_simulation::advect_scalar_kernel, 0, 0) != cudaSuccess) return dim3(8u, 8u, 4u);
            if (block_size <= 0) return dim3(8u, 8u, 4u);
            unsigned block_z = block_size >= 256 ? 4u : block_size >= 128 ? 2u : 1u;
            unsigned block_y = block_size / static_cast<int>(block_z) >= 64 ? 8u : block_size / static_cast<int>(block_z) >= 32 ? 4u : 2u;
            unsigned block_x = static_cast<unsigned>((std::max) (block_size / static_cast<int>(block_y * block_z), 1));
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
        context->cells            = dim3(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));
        context->velocity_x_cells = dim3(static_cast<unsigned>(((context->config.nx + 1) + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));
        context->velocity_y_cells = dim3(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>(((context->config.ny + 1) + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));
        context->velocity_z_cells = dim3(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), static_cast<unsigned>(((context->config.nz + 1) + static_cast<int>(context->block.z) - 1) / static_cast<int>(context->block.z)));

        context->device.scalar_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::ScalarField{.kind = smoke_simulation::SMOKE_FIELD_DENSITY});
        context->device.scalar_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::ScalarField{.kind = smoke_simulation::SMOKE_FIELD_TEMPERATURE});
        context->device.vector_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::VectorField{.kind = smoke_simulation::SMOKE_VECTOR_FORCE});
        context->device.vector_fields.push_back(smoke_simulation::ContextStorage::DeviceBuffers::VectorField{.kind = smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY});

        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_x), context->velocity_x_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_y), context->velocity_y_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_z), context->velocity_z_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_x), context->velocity_x_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_y), context->velocity_y_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.temp_velocity_z), context->velocity_z_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.centered_velocity_x), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.centered_velocity_y), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.centered_velocity_z), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.velocity_magnitude), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.pressure), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.pressure_rhs), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.divergence), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.vorticity_x), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.vorticity_y), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.vorticity_z), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.vorticity_magnitude), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.force_x), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.force_y), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.flow.force_z), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.occupancy_float), context->cell_bytes), "cudaMalloc float");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.occupancy), context->cell_count * sizeof(uint8_t)), "cudaMalloc uint8_t");
        smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&context->device.solid_temperature), context->cell_bytes), "cudaMalloc float");
        for (auto& field : context->device.scalar_fields) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.data), context->cell_bytes), "cudaMalloc float");
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.temp), context->cell_bytes), "cudaMalloc float");
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.source), context->cell_bytes), "cudaMalloc float");
        }
        for (auto& field : context->device.vector_fields) {
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.data_x), context->cell_bytes), "cudaMalloc float");
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.data_y), context->cell_bytes), "cudaMalloc float");
            smoke_simulation::check_cuda(cudaMalloc(reinterpret_cast<void**>(&field.data_z), context->cell_bytes), "cudaMalloc float");
        }

        const auto linear_grid = static_cast<unsigned>((context->cell_count + 255u) / 256u);
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
        for (auto& [kind, data, temp, source] : context->device.scalar_fields) {
            const float initial_value = kind == smoke_simulation::SMOKE_FIELD_DENSITY ? desc->initial_density : desc->initial_temperature;
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(data, initial_value, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(temp, initial_value, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(source, 0.0f, context->cell_count);
        }
        for (auto& field : context->device.vector_fields) {
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_x, 0.0f, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_y, 0.0f, context->cell_count);
            smoke_simulation::fill_float_kernel<<<linear_grid, 256, 0, context->stream>>>(field.data_z, 0.0f, context->cell_count);
        }
        smoke_simulation::check_cuda(cudaGetLastError(), "create_context init");
        smoke_simulation::initialize_pressure_system(*context);

        constexpr dim3 linear_block(256u, 1u, 1u);
        const dim3 linear_cells(static_cast<unsigned>((context->cell_count + 255u) / 256u), 1u, 1u);
        const dim3 sync_block(context->block.x, context->block.y, 1u);
        const dim3 sync_velocity_x_grid(static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), 1u);
        const dim3 sync_velocity_y_grid(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.nz + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), 1u);
        const dim3 sync_velocity_z_grid(static_cast<unsigned>((context->config.nx + static_cast<int>(context->block.x) - 1) / static_cast<int>(context->block.x)), static_cast<unsigned>((context->config.ny + static_cast<int>(context->block.y) - 1) / static_cast<int>(context->block.y)), 1u);
        const bool periodic_x = context->config.flow_boundary.x_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context->config.flow_boundary.x_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_y = context->config.flow_boundary.y_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context->config.flow_boundary.y_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;
        const bool periodic_z = context->config.flow_boundary.z_minus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC && context->config.flow_boundary.z_plus.type == SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC;

        smoke_simulation::check_cuda(cudaStreamBeginCapture(context->stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture step");
        smoke_simulation::apply_solid_temperature_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data, context->device.occupancy, context->device.solid_temperature, context->config.nx, context->config.ny, context->config.nz, context->config.ambient_temperature);
        smoke_simulation::compute_center_velocity_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.centered_velocity_x, context->device.flow.centered_velocity_y, context->device.flow.centered_velocity_z, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->config.nx, context->config.ny, context->config.nz);
        smoke_simulation::compute_vorticity_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.vorticity_x, context->device.flow.vorticity_y, context->device.flow.vorticity_z, context->device.flow.vorticity_magnitude, context->device.flow.centered_velocity_x, context->device.flow.centered_velocity_y, context->device.flow.centered_velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.flow_boundary);
        smoke_simulation::seed_force_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.flow.force_x, context->device.flow.force_y, context->device.flow.force_z, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE].data_x, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE].data_y, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE].data_z, context->cell_count);
        smoke_simulation::add_buoyancy_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.force_y, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.ambient_temperature, context->config.buoyancy_density_factor, context->config.buoyancy_temperature_factor, context->config.flow_boundary);
        smoke_simulation::add_confinement_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.force_x, context->device.flow.force_y, context->device.flow.force_z, context->device.flow.vorticity_x, context->device.flow.vorticity_y, context->device.flow.vorticity_z, context->device.flow.vorticity_magnitude, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.vorticity_confinement, context->config.flow_boundary);
        smoke_simulation::add_center_forces_to_velocity_x_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_x, context->device.flow.force_x, context->config.nx, context->config.ny, context->config.nz, context->config.dt);
        smoke_simulation::add_center_forces_to_velocity_y_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_y, context->device.flow.force_y, context->config.nx, context->config.ny, context->config.nz, context->config.dt);
        smoke_simulation::add_center_forces_to_velocity_z_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_z, context->device.flow.force_z, context->config.nx, context->config.ny, context->config.nz, context->config.dt);
        smoke_simulation::enforce_velocity_x_boundaries_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_x, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_x, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_y_boundaries_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_y, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_y, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_z_boundaries_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.velocity_z, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_z, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        if (periodic_x) smoke_simulation::sync_periodic_velocity_x_kernel<<<sync_velocity_x_grid, sync_block, 0, context->stream>>>(context->device.flow.velocity_x, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_y) smoke_simulation::sync_periodic_velocity_y_kernel<<<sync_velocity_y_grid, sync_block, 0, context->stream>>>(context->device.flow.velocity_y, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_z) smoke_simulation::sync_periodic_velocity_z_kernel<<<sync_velocity_z_grid, sync_block, 0, context->stream>>>(context->device.flow.velocity_z, context->config.nx, context->config.ny, context->config.nz);
        smoke_simulation::advect_velocity_x_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->device.flow.velocity_x, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.scalar_advection_mode, context->config.flow_boundary);
        smoke_simulation::advect_velocity_y_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->device.flow.velocity_y, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.scalar_advection_mode, context->config.flow_boundary);
        smoke_simulation::advect_velocity_z_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->device.flow.velocity_z, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.scalar_advection_mode, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_x_boundaries_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_x, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_y_boundaries_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_y, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_z_boundaries_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_z, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        if (periodic_x) smoke_simulation::sync_periodic_velocity_x_kernel<<<sync_velocity_x_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_y) smoke_simulation::sync_periodic_velocity_y_kernel<<<sync_velocity_y_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_z) smoke_simulation::sync_periodic_velocity_z_kernel<<<sync_velocity_z_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->config.nx, context->config.ny, context->config.nz);
        smoke_simulation::solve_pressure_pcg(*context);
        smoke_simulation::project_velocity_x_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->device.flow.pressure, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_x, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.flow_boundary);
        smoke_simulation::project_velocity_y_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->device.flow.pressure, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_y, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.flow_boundary);
        smoke_simulation::project_velocity_z_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->device.flow.pressure, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_z, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_x_boundaries_kernel<<<context->velocity_x_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_x, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_y_boundaries_kernel<<<context->velocity_y_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_y, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        smoke_simulation::enforce_velocity_z_boundaries_kernel<<<context->velocity_z_cells, context->block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->device.occupancy, context->device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY].data_z, context->config.nx, context->config.ny, context->config.nz, context->config.flow_boundary);
        if (periodic_x) smoke_simulation::sync_periodic_velocity_x_kernel<<<sync_velocity_x_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_x, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_y) smoke_simulation::sync_periodic_velocity_y_kernel<<<sync_velocity_y_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_y, context->config.nx, context->config.ny, context->config.nz);
        if (periodic_z) smoke_simulation::sync_periodic_velocity_z_kernel<<<sync_velocity_z_grid, sync_block, 0, context->stream>>>(context->device.flow.temp_velocity_z, context->config.nx, context->config.ny, context->config.nz);
        smoke_simulation::check_cuda(cudaMemcpyAsync(context->device.flow.velocity_x, context->device.flow.temp_velocity_x, context->velocity_x_bytes, cudaMemcpyDeviceToDevice, context->stream), "cudaMemcpyAsync velocity_x");
        smoke_simulation::check_cuda(cudaMemcpyAsync(context->device.flow.velocity_y, context->device.flow.temp_velocity_y, context->velocity_y_bytes, cudaMemcpyDeviceToDevice, context->stream), "cudaMemcpyAsync velocity_y");
        smoke_simulation::check_cuda(cudaMemcpyAsync(context->device.flow.velocity_z, context->device.flow.temp_velocity_z, context->velocity_z_bytes, cudaMemcpyDeviceToDevice, context->stream), "cudaMemcpyAsync velocity_z");
        smoke_simulation::add_source_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].temp, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].source, context->config.dt, context->cell_count);
        smoke_simulation::advect_scalar_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].temp, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.scalar_advection_mode, context->config.temperature_boundary, context->config.flow_boundary);
        smoke_simulation::apply_solid_temperature_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE].data, context->device.occupancy, context->device.solid_temperature, context->config.nx, context->config.ny, context->config.nz, context->config.ambient_temperature);
        smoke_simulation::add_source_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].temp, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].source, context->config.dt, context->cell_count);
        smoke_simulation::advect_scalar_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].temp, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.dt, context->config.scalar_advection_mode, context->config.density_boundary, context->config.flow_boundary);
        smoke_simulation::boundary_fill_density_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].temp, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.density_boundary);
        smoke_simulation::check_cuda(cudaMemcpyAsync(context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].data, context->device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY].temp, context->cell_bytes, cudaMemcpyDeviceToDevice, context->stream), "cudaMemcpyAsync density");
        smoke_simulation::compute_center_velocity_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.centered_velocity_x, context->device.flow.centered_velocity_y, context->device.flow.centered_velocity_z, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->config.nx, context->config.ny, context->config.nz);
        smoke_simulation::compute_vorticity_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.vorticity_x, context->device.flow.vorticity_y, context->device.flow.vorticity_z, context->device.flow.vorticity_magnitude, context->device.flow.centered_velocity_x, context->device.flow.centered_velocity_y, context->device.flow.centered_velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size, context->config.flow_boundary);
        smoke_simulation::compute_divergence_kernel<<<context->cells, context->block, 0, context->stream>>>(context->device.flow.divergence, context->device.flow.velocity_x, context->device.flow.velocity_y, context->device.flow.velocity_z, context->device.occupancy, context->config.nx, context->config.ny, context->config.nz, context->config.cell_size);
        smoke_simulation::velocity_magnitude_kernel<<<linear_cells, linear_block, 0, context->stream>>>(context->device.flow.velocity_magnitude, context->device.flow.centered_velocity_x, context->device.flow.centered_velocity_y, context->device.flow.centered_velocity_z, context->cell_count);
        smoke_simulation::check_cuda(cudaStreamEndCapture(context->stream, &context->step_graph.graph), "cudaStreamEndCapture step");
        smoke_simulation::check_cuda(cudaGraphInstantiate(&context->step_graph.exec, context->step_graph.graph), "cudaGraphInstantiate step");

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
    auto& storage       = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& density_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY];
    if (values == nullptr) return cudaMemsetAsync(density_field.data, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(density_field.data, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_density_source_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_density_source");
    auto& storage       = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& density_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_DENSITY];
    if (values == nullptr) return cudaMemsetAsync(density_field.source, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(density_field.source, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_temperature_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_temperature");
    auto& storage           = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& temperature_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE];
    if (values == nullptr) return cudaMemsetAsync(temperature_field.data, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(temperature_field.data, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_temperature_source_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_temperature_source");
    auto& storage           = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& temperature_field = storage.device.scalar_fields[smoke_simulation::SMOKE_FIELD_TEMPERATURE];
    if (values == nullptr) return cudaMemsetAsync(temperature_field.source, 0, storage.cell_bytes, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return cudaMemcpyAsync(temperature_field.source, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_update_force_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z) {
    nvtx3::scoped_range range("smoke.update_force");
    auto& storage     = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& force_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_FORCE];
    if ((values_x == nullptr ? cudaMemsetAsync(force_field.data_x, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_x, values_x, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_y == nullptr ? cudaMemsetAsync(force_field.data_y, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_y, values_y, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_z == nullptr ? cudaMemsetAsync(force_field.data_z, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(force_field.data_z, values_z, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_occupancy_cuda(SmokeSimulationContext context, const uint8_t* values) {
    nvtx3::scoped_range range("smoke.update_occupancy");
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);
    if (values == nullptr) {
        if (cudaMemsetAsync(storage.device.occupancy, 0, storage.cell_count * sizeof(uint8_t), storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        if (cudaMemsetAsync(storage.device.occupancy_float, 0, storage.cell_bytes, storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
        return SMOKE_SIMULATION_RESULT_OK;
    }
    if (cudaMemcpyAsync(storage.device.occupancy, values, storage.cell_count * sizeof(uint8_t), cudaMemcpyDeviceToDevice, storage.stream) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    smoke_simulation::copy_u8_to_float_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.occupancy_float, storage.device.occupancy, storage.cell_count);
    if (cudaGetLastError() != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_solid_velocity_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z) {
    nvtx3::scoped_range range("smoke.update_solid_velocity");
    auto& storage              = *static_cast<smoke_simulation::ContextStorage*>(context);
    auto& solid_velocity_field = storage.device.vector_fields[smoke_simulation::SMOKE_VECTOR_SOLID_VELOCITY];
    if ((values_x == nullptr ? cudaMemsetAsync(solid_velocity_field.data_x, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_x, values_x, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_y == nullptr ? cudaMemsetAsync(solid_velocity_field.data_y, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_y, values_y, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    if ((values_z == nullptr ? cudaMemsetAsync(solid_velocity_field.data_z, 0, storage.cell_bytes, storage.stream) : cudaMemcpyAsync(solid_velocity_field.data_z, values_z, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream)) != cudaSuccess) return SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    return SMOKE_SIMULATION_RESULT_OK;
}

SmokeSimulationResult smoke_simulation_update_solid_temperature_cuda(SmokeSimulationContext context, const float* values) {
    nvtx3::scoped_range range("smoke.update_solid_temperature");
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);
    if (values == nullptr) {
        smoke_simulation::fill_float_kernel<<<static_cast<unsigned>((storage.cell_count + 255u) / 256u), 256, 0, storage.stream>>>(storage.device.solid_temperature, storage.config.ambient_temperature, storage.cell_count);
        return cudaGetLastError() == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
    }
    return cudaMemcpyAsync(storage.device.solid_temperature, values, storage.cell_bytes, cudaMemcpyDeviceToDevice, storage.stream) == cudaSuccess ? SMOKE_SIMULATION_RESULT_OK : SMOKE_SIMULATION_RESULT_BACKEND_FAILURE;
}

SmokeSimulationResult smoke_simulation_step_cuda(SmokeSimulationContext context) {
    nvtx3::scoped_range range("smoke.step");
    auto& storage = *static_cast<smoke_simulation::ContextStorage*>(context);

    try {
        smoke_simulation::check_cuda(cudaGraphLaunch(storage.step_graph.exec, storage.stream), "cudaGraphLaunch step");
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
