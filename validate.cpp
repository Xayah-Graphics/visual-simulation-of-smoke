#include "visual-simulation-of-smoke.h"

extern "C" {

namespace {

    int32_t validate_base(const uint32_t struct_size, const uint32_t expected_size, const uint32_t api_version) {
        if (struct_size < expected_size) return 1000;
        if (api_version != VISUAL_SIMULATION_OF_SMOKE_API_VERSION) return 1006;
        return 0;
    }

    int32_t validate_grid(const int32_t nx, const int32_t ny, const int32_t nz, const float cell_size, const float dt) {
        if (nx <= 0 || ny <= 0 || nz <= 0) return 1001;
        if (cell_size <= 0.0f) return 1002;
        if (dt <= 0.0f) return 1003;
        return 0;
    }

    int32_t validate_boundaries(const uint32_t boundary_x_min, const uint32_t boundary_x_max, const uint32_t boundary_y_min, const uint32_t boundary_y_max, const uint32_t boundary_z_min, const uint32_t boundary_z_max) {
        if (boundary_x_min > VISUAL_SMOKE_BOUNDARY_OUTFLOW || boundary_x_max > VISUAL_SMOKE_BOUNDARY_OUTFLOW || boundary_y_min > VISUAL_SMOKE_BOUNDARY_OUTFLOW || boundary_y_max > VISUAL_SMOKE_BOUNDARY_OUTFLOW || boundary_z_min > VISUAL_SMOKE_BOUNDARY_OUTFLOW || boundary_z_max > VISUAL_SMOKE_BOUNDARY_OUTFLOW) return 1005;
        return 0;
    }

} // namespace

int32_t visual_simulation_of_smoke_validate_forces_desc(const VisualSimulationOfSmokeForcesDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeForcesDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->density == nullptr) return 2001;
    if (desc->temperature == nullptr) return 2002;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_omega_x == nullptr) return 2014;
    if (desc->temporary_omega_y == nullptr) return 2015;
    if (desc->temporary_omega_z == nullptr) return 2016;
    if (desc->temporary_omega_magnitude == nullptr) return 2017;
    if (desc->temporary_force_x == nullptr) return 2018;
    if (desc->temporary_force_y == nullptr) return 2019;
    if (desc->temporary_force_z == nullptr) return 2020;
    return 0;
}

int32_t visual_simulation_of_smoke_validate_advect_velocity_desc(const VisualSimulationOfSmokeAdvectVelocityDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeAdvectVelocityDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    if (desc->temporary_previous_velocity_x == nullptr) return 2009;
    if (desc->temporary_previous_velocity_y == nullptr) return 2010;
    if (desc->temporary_previous_velocity_z == nullptr) return 2011;
    return 0;
}

int32_t visual_simulation_of_smoke_validate_project_desc(const VisualSimulationOfSmokeProjectDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeProjectDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (desc->pressure_iterations <= 0) return 1004;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->temporary_previous_velocity_x == nullptr) return 2009;
    if (desc->temporary_previous_velocity_y == nullptr) return 2010;
    if (desc->temporary_previous_velocity_z == nullptr) return 2011;
    if (desc->temporary_pressure == nullptr) return 2012;
    if (desc->temporary_divergence == nullptr) return 2013;
    if (desc->temporary_omega_x == nullptr) return 2014;
    if (desc->temporary_omega_y == nullptr) return 2015;
    return 0;
}


int32_t visual_simulation_of_smoke_validate_advect_scalar_flow_desc(const VisualSimulationOfSmokeAdvectScalarFlowDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeAdvectScalarFlowDesc), desc->api_version); code != 0) return code;
    if (const int32_t code = validate_grid(desc->nx, desc->ny, desc->nz, desc->cell_size, desc->dt); code != 0) return code;
    if (const int32_t code = validate_boundaries(desc->boundary_x_min, desc->boundary_x_max, desc->boundary_y_min, desc->boundary_y_max, desc->boundary_z_min, desc->boundary_z_max); code != 0) return code;
    if (desc->scalar_bindings == nullptr || desc->scalar_count <= 0) return 1000;
    if (desc->velocity_x == nullptr) return 2003;
    if (desc->velocity_y == nullptr) return 2004;
    if (desc->velocity_z == nullptr) return 2005;
    for (int32_t i = 0; i < desc->scalar_count; ++i) {
        if (desc->scalar_bindings[i].scalar == nullptr) return 2001;
        if (desc->scalar_bindings[i].temporary_previous_scalar == nullptr) return 2007;
    }
    return 0;
}

int32_t visual_simulation_of_smoke_validate_add_scalar_source_desc(const VisualSimulationOfSmokeAddScalarSourceDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeAddScalarSourceDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->scalar == nullptr) return 2001;
    return 0;
}

int32_t visual_simulation_of_smoke_validate_add_vector_source_desc(const VisualSimulationOfSmokeAddVectorSourceDesc* desc) {
    if (desc == nullptr) return 1000;
    if (const int32_t code = validate_base(desc->struct_size, sizeof(VisualSimulationOfSmokeAddVectorSourceDesc), desc->api_version); code != 0) return code;
    if (desc->nx <= 0 || desc->ny <= 0 || desc->nz <= 0) return 1001;
    if (desc->vector_x == nullptr) return 2003;
    if (desc->vector_y == nullptr) return 2004;
    if (desc->vector_z == nullptr) return 2005;
    return 0;
}

} // extern "C"
