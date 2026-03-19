#ifndef VISUAL_SIMULATION_OF_SMOKE_H
#define VISUAL_SIMULATION_OF_SMOKE_H

#include <stdint.h>

#ifndef SMOKE_FIELD_TYPES_DEFINED
#define SMOKE_FIELD_TYPES_DEFINED

#define FIELD_FORMAT_F32 1u
#define FIELD_MEMORY_TYPE_CUDA_DEVICE 1u

typedef struct FieldGridDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
} FieldGridDesc;

typedef struct FieldBufferView {
    void* data;
    uint64_t size_bytes;
    uint32_t format;
    uint32_t memory_type;
} FieldBufferView;

typedef struct ScalarField {
    FieldGridDesc grid;
    FieldBufferView values;
} ScalarField;

typedef enum VectorFieldLayout {
    VECTOR_FIELD_LAYOUT_CELL_CENTERED = 1,
    VECTOR_FIELD_LAYOUT_STAGGERED_MAC = 2,
} VectorFieldLayout;

typedef enum VectorFieldComponent {
    VECTOR_FIELD_COMPONENT_X = 0,
    VECTOR_FIELD_COMPONENT_Y = 1,
    VECTOR_FIELD_COMPONENT_Z = 2,
} VectorFieldComponent;

typedef struct VectorField {
    FieldGridDesc grid;
    uint32_t layout;
    FieldBufferView x;
    FieldBufferView y;
    FieldBufferView z;
} VectorField;

#endif

#if defined(_WIN32)
#if defined(VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED)
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllexport)
#else
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define VISUAL_SIMULATION_OF_SMOKE_API __attribute__((visibility("default")))
#else
#define VISUAL_SIMULATION_OF_SMOKE_API
#endif

#define VISUAL_SIMULATION_OF_SMOKE_SUCCESS                 0
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_INVALID_ARGUMENT  -1
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_RUNTIME           -2
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_ALLOCATION_FAILED -3
#define VISUAL_SIMULATION_OF_SMOKE_ERROR_BUFFER_TOO_SMALL  -4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisualSimulationOfSmokeContextDesc {
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float dt;
    float cell_size;
    float ambient_temperature;
    float density_buoyancy;
    float temperature_buoyancy;
    float vorticity_epsilon;
    int32_t pressure_iterations;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    uint32_t use_monotonic_cubic;
} VisualSimulationOfSmokeContextDesc;

typedef struct VisualSimulationOfSmokeSourceDesc {
    float center_x;
    float center_y;
    float center_z;
    float radius;
    float density_amount;
    float temperature_amount;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} VisualSimulationOfSmokeSourceDesc;

typedef struct VisualSimulationOfSmokeContext VisualSimulationOfSmokeContext;

VISUAL_SIMULATION_OF_SMOKE_API VisualSimulationOfSmokeContextDesc visual_simulation_of_smoke_context_desc_default(void);
VISUAL_SIMULATION_OF_SMOKE_API VisualSimulationOfSmokeContext* visual_simulation_of_smoke_context_create(const VisualSimulationOfSmokeContextDesc* desc);
VISUAL_SIMULATION_OF_SMOKE_API void visual_simulation_of_smoke_context_destroy(VisualSimulationOfSmokeContext* context);

VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_context_required_scalar_field_bytes(const VisualSimulationOfSmokeContext* context);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_context_required_vector_field_component_bytes(const VisualSimulationOfSmokeContext* context, uint32_t component);

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_clear_async(VisualSimulationOfSmokeContext* context, void* cuda_stream);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_add_source_async(VisualSimulationOfSmokeContext* context, const VisualSimulationOfSmokeSourceDesc* source, void* cuda_stream);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_step_async(VisualSimulationOfSmokeContext* context, void* cuda_stream);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_snapshot_density_async(VisualSimulationOfSmokeContext* context, const ScalarField* destination, void* cuda_stream);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_snapshot_temperature_async(VisualSimulationOfSmokeContext* context, const ScalarField* destination, void* cuda_stream);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_snapshot_velocity_magnitude_async(VisualSimulationOfSmokeContext* context, const ScalarField* destination, void* cuda_stream);

VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_last_error_length(void);
VISUAL_SIMULATION_OF_SMOKE_API uint64_t visual_simulation_of_smoke_context_last_error_length(const VisualSimulationOfSmokeContext* context);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_copy_last_error(char* buffer, uint64_t buffer_size);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_copy_context_last_error(const VisualSimulationOfSmokeContext* context, char* buffer, uint64_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // VISUAL_SIMULATION_OF_SMOKE_H
