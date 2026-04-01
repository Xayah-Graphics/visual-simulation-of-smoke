#ifndef VISUAL_SIMULATION_OF_SMOKE_3D_H
#define VISUAL_SIMULATION_OF_SMOKE_3D_H

#include <stdint.h>

#ifdef _WIN32
#ifdef VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED
#define SMOKE_SIMULATION_API __declspec(dllexport)
#else
#define SMOKE_SIMULATION_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define SMOKE_SIMULATION_API __attribute__((visibility("default")))
#else
#define SMOKE_SIMULATION_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SmokeSimulationResult {
    SMOKE_SIMULATION_RESULT_OK              = 0,
    SMOKE_SIMULATION_RESULT_OUT_OF_MEMORY   = 1,
    SMOKE_SIMULATION_RESULT_BACKEND_FAILURE = 2,
} SmokeSimulationResult;

typedef enum SmokeSimulationFlowBoundaryType {
    SMOKE_SIMULATION_FLOW_BOUNDARY_NO_SLIP_WALL   = 0,
    SMOKE_SIMULATION_FLOW_BOUNDARY_FREE_SLIP_WALL = 1,
    SMOKE_SIMULATION_FLOW_BOUNDARY_OUTFLOW        = 2,
    SMOKE_SIMULATION_FLOW_BOUNDARY_PERIODIC       = 3,
} SmokeSimulationFlowBoundaryType;

typedef enum SmokeSimulationScalarBoundaryType {
    SMOKE_SIMULATION_SCALAR_BOUNDARY_FIXED_VALUE = 0,
    SMOKE_SIMULATION_SCALAR_BOUNDARY_ZERO_FLUX   = 1,
    SMOKE_SIMULATION_SCALAR_BOUNDARY_PERIODIC    = 2,
} SmokeSimulationScalarBoundaryType;

typedef struct SmokeSimulationFlowBoundaryFaceDesc {
    uint32_t type;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float pressure;
} SmokeSimulationFlowBoundaryFaceDesc;

typedef struct SmokeSimulationFlowBoundaryConfig {
    SmokeSimulationFlowBoundaryFaceDesc x_minus;
    SmokeSimulationFlowBoundaryFaceDesc x_plus;
    SmokeSimulationFlowBoundaryFaceDesc y_minus;
    SmokeSimulationFlowBoundaryFaceDesc y_plus;
    SmokeSimulationFlowBoundaryFaceDesc z_minus;
    SmokeSimulationFlowBoundaryFaceDesc z_plus;
} SmokeSimulationFlowBoundaryConfig;

typedef struct SmokeSimulationScalarBoundaryFaceDesc {
    uint32_t type;
    float value;
} SmokeSimulationScalarBoundaryFaceDesc;

typedef struct SmokeSimulationScalarBoundaryConfig {
    SmokeSimulationScalarBoundaryFaceDesc x_minus;
    SmokeSimulationScalarBoundaryFaceDesc x_plus;
    SmokeSimulationScalarBoundaryFaceDesc y_minus;
    SmokeSimulationScalarBoundaryFaceDesc y_plus;
    SmokeSimulationScalarBoundaryFaceDesc z_minus;
    SmokeSimulationScalarBoundaryFaceDesc z_plus;
} SmokeSimulationScalarBoundaryConfig;

typedef enum SmokeSimulationScalarAdvectionMode {
    SMOKE_SIMULATION_SCALAR_ADVECTION_LINEAR           = 0,
    SMOKE_SIMULATION_SCALAR_ADVECTION_MONOTONIC_CUBIC = 1,
} SmokeSimulationScalarAdvectionMode;

typedef struct SmokeSimulationConfig {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    int32_t pressure_iterations;
    float ambient_temperature;
    float buoyancy_density_factor;
    float buoyancy_temperature_factor;
    float vorticity_confinement;
    SmokeSimulationScalarAdvectionMode scalar_advection_mode;
    SmokeSimulationFlowBoundaryConfig flow_boundary;
    SmokeSimulationScalarBoundaryConfig density_boundary;
    SmokeSimulationScalarBoundaryConfig temperature_boundary;
} SmokeSimulationConfig;

typedef struct SmokeSimulationContext_t* SmokeSimulationContext;

typedef struct SmokeSimulationContextCreateDesc {
    SmokeSimulationConfig config;
    void* stream;
    float initial_density;
    float initial_temperature;
} SmokeSimulationContextCreateDesc;

SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_create_context_cuda(const SmokeSimulationContextCreateDesc* desc, SmokeSimulationContext* out_context);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_destroy_context_cuda(SmokeSimulationContext context);

SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_density_cuda(SmokeSimulationContext context, const float* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_density_source_cuda(SmokeSimulationContext context, const float* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_temperature_cuda(SmokeSimulationContext context, const float* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_temperature_source_cuda(SmokeSimulationContext context, const float* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_force_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_occupancy_cuda(SmokeSimulationContext context, const uint8_t* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_solid_velocity_cuda(SmokeSimulationContext context, const float* values_x, const float* values_y, const float* values_z);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_update_solid_temperature_cuda(SmokeSimulationContext context, const float* values);
SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_step_cuda(SmokeSimulationContext context);

typedef enum SmokeSimulationViewKind {
    SMOKE_SIMULATION_VIEW_DENSITY              = 0,
    SMOKE_SIMULATION_VIEW_DENSITY_SOURCE       = 1,
    SMOKE_SIMULATION_VIEW_TEMPERATURE          = 2,
    SMOKE_SIMULATION_VIEW_TEMPERATURE_SOURCE   = 3,
    SMOKE_SIMULATION_VIEW_FORCE                = 4,
    SMOKE_SIMULATION_VIEW_SOLID_VELOCITY       = 5,
    SMOKE_SIMULATION_VIEW_SOLID_TEMPERATURE    = 6,
    SMOKE_SIMULATION_VIEW_FLOW_VELOCITY        = 7,
    SMOKE_SIMULATION_VIEW_FLOW_VELOCITY_MAGNITUDE = 8,
    SMOKE_SIMULATION_VIEW_FLOW_PRESSURE        = 9,
    SMOKE_SIMULATION_VIEW_FLOW_PRESSURE_RHS    = 10,
    SMOKE_SIMULATION_VIEW_FLOW_DIVERGENCE      = 11,
    SMOKE_SIMULATION_VIEW_FLOW_VORTICITY       = 12,
    SMOKE_SIMULATION_VIEW_FLOW_VORTICITY_MAGNITUDE = 13,
    SMOKE_SIMULATION_VIEW_OCCUPANCY            = 14,
} SmokeSimulationViewKind;

typedef enum SmokeSimulationViewLayout {
    SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D      = 0,
    SMOKE_SIMULATION_VIEW_LAYOUT_F32_3D_SOA3 = 1,
} SmokeSimulationViewLayout;

typedef struct SmokeSimulationViewRequest {
    uint32_t kind;
    void* consumer_stream;
} SmokeSimulationViewRequest;

typedef struct SmokeSimulationView {
    uint32_t layout;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    uint64_t row_stride_bytes;
    uint64_t slice_stride_bytes;
    const float* data0;
    const float* data1;
    const float* data2;
} SmokeSimulationView;

SMOKE_SIMULATION_API SmokeSimulationResult smoke_simulation_get_view_cuda(SmokeSimulationContext context, const SmokeSimulationViewRequest* request, SmokeSimulationView* out_view);

#ifdef __cplusplus
}
#endif

#endif // VISUAL_SIMULATION_OF_SMOKE_3D_H
