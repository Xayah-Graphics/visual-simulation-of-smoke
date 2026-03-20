#ifndef VISUAL_SIMULATION_OF_SMOKE_H
#define VISUAL_SIMULATION_OF_SMOKE_H

#include <stdint.h>

#ifdef _WIN32
#ifdef VISUAL_SIMULATION_OF_SMOKE_BUILD_SHARED
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllexport)
#else
#define VISUAL_SIMULATION_OF_SMOKE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define VISUAL_SIMULATION_OF_SMOKE_API __attribute__((visibility("default")))
#else
#define VISUAL_SIMULATION_OF_SMOKE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
Error code scheme:
0     : success
1xxx  : scalar/grid/step parameter errors
1000  : invalid step descriptor
1001  : invalid grid dimensions
1002  : invalid cell size
1003  : invalid dt
1004  : invalid iteration count
2xxx  : buffer errors
2001  : invalid density buffer
2002  : invalid temperature buffer
2003  : invalid velocity_x buffer
2004  : invalid velocity_y buffer
2005  : invalid velocity_z buffer
2007  : invalid temporary previous density buffer
2008  : invalid temporary previous temperature buffer
2009  : invalid temporary previous velocity_x buffer
2010  : invalid temporary previous velocity_y buffer
2011  : invalid temporary previous velocity_z buffer
2012  : invalid temporary pressure buffer
2013  : invalid temporary divergence buffer
2014  : invalid temporary omega_x buffer
2015  : invalid temporary omega_y buffer
2016  : invalid temporary omega_z buffer
2017  : invalid temporary omega_magnitude buffer
2018  : invalid temporary force_x buffer
2019  : invalid temporary force_y buffer
2020  : invalid temporary force_z buffer
5xxx  : CUDA runtime or kernel launch failure
5001  : CUDA call failed
*/

typedef struct VisualSimulationOfSmokeStepDesc {
    uint32_t struct_size;
    uint32_t api_version;
    int32_t nx;
    int32_t ny;
    int32_t nz;
    float cell_size;
    float dt;
    float ambient_temperature;
    float density_buoyancy;
    float temperature_buoyancy;
    float vorticity_epsilon;
    int32_t pressure_iterations;
    uint32_t use_monotonic_cubic;
    void* density;
    void* temperature;
    void* velocity_x;
    void* velocity_y;
    void* velocity_z;
    void* temporary_previous_density;
    void* temporary_previous_temperature;
    void* temporary_previous_velocity_x;
    void* temporary_previous_velocity_y;
    void* temporary_previous_velocity_z;
    void* temporary_pressure;
    void* temporary_divergence;
    void* temporary_omega_x;
    void* temporary_omega_y;
    void* temporary_omega_z;
    void* temporary_omega_magnitude;
    void* temporary_force_x;
    void* temporary_force_y;
    void* temporary_force_z;
    int32_t block_x;
    int32_t block_y;
    int32_t block_z;
    void* stream;
} VisualSimulationOfSmokeStepDesc;

VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_validate_desc(const VisualSimulationOfSmokeStepDesc* desc);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_step_cuda(const VisualSimulationOfSmokeStepDesc* desc);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_step_cpu(const VisualSimulationOfSmokeStepDesc* desc);
VISUAL_SIMULATION_OF_SMOKE_API int32_t visual_simulation_of_smoke_step_parallel(const VisualSimulationOfSmokeStepDesc* desc);

#ifdef __cplusplus
}
#endif

#endif // VISUAL_SIMULATION_OF_SMOKE_H
