import pytest
from mush3p import NonDimensionalParams, solve

# Test parameters for the air-water-ice system
NON_DIM_PARAMS = {
    "concentration_ratio": 0.13,
    "stefan_number": 4.2,
    "far_temperature_scaled": 0.14,
    "expansion_coefficient": 0.029,
    "stokes_rise_velocity_scaled": 4.6e4,
    "pore_throat_exponent": 0.46,
    "solid_conductivity_ratio": 2,
    "solid_specific_heat_capacity_ratio": 0.5,
    "gas_specific_heat_capacity_ratio": 0.25,
    "gas_conductivity_ratio": 0.05,
    "gas_density_ratio": 1e-3,
    "damkholer_number": 100,
    "far_dissolved_concentration_scaled": 1,
    "hele_shaw_permeability_scaled": 210,
    "hydrostatic_pressure_scale": 0.012,
    "laplace_pressure_scale": 0.0015,
    "kelvin_conversion_temperature": 14.0,
    "atmospheric_pressure_scaled": 3e6,
}


@pytest.mark.parametrize("solver", ["full", "incompressible", "reduced", "instant"])
def test_solve_immobile_bubble_parameters(solver: str) -> None:
    params = NonDimensionalParams(
        name=solver,
        model_choice=solver,
        bubble_radius_scaled=1,
        **NON_DIM_PARAMS,
    )
    # This will raise an error if the solver fails to converge
    solve(params)


@pytest.mark.parametrize("solver", ["full", "incompressible", "reduced", "instant"])
def test_solve_mobile_bubble_parameters(solver: str) -> None:
    params = NonDimensionalParams(
        name=solver,
        model_choice=solver,
        bubble_radius_scaled=0.1,
        **NON_DIM_PARAMS,
    )
    # This will raise an error if the solver fails to converge
    solve(params)
