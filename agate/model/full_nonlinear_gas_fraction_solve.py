import numpy as np
from ..params import NonDimensionalParams
from ..static_settings import HARTHOLT_DRAG_FUNCTION


def calculate_liquid_darcy_velocity(gas_fraction, frozen_gas_fraction):
    return gas_fraction - frozen_gas_fraction


def calculate_liquid_fraction(gas_fraction, solid_fraction):
    return 1 - solid_fraction - gas_fraction


def calculate_bubble_radius(solid_fraction, non_dimensional_params):
    exponent = non_dimensional_params.pore_throat_exponent
    return non_dimensional_params.bubble_radius_scaled / (
        (1 - solid_fraction) ** exponent
    )


def calculate_lag(bubble_radius):
    lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
    lag = np.where(bubble_radius > 1, 0.5, lag)
    return lag


def calculate_drag(bubble_radius):
    drag = np.where(bubble_radius < 0, 1, HARTHOLT_DRAG_FUNCTION(bubble_radius))
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_darcy_velocity(solid_fraction, gas_fraction, non_dimensional_params):
    bubble_scale = non_dimensional_params.bubble_radius_scaled
    drag = calculate_drag(
        calculate_bubble_radius(solid_fraction, non_dimensional_params)
    )

    return (
        gas_fraction
        * non_dimensional_params.stokes_rise_velocity_scaled
        * drag
        * bubble_scale**2
    )


def calculate_gas_density(
    height, mushy_layer_depth, temperature, hydrostatic_pressure, non_dimensional_params
):
    kelvin = non_dimensional_params.kelvin_conversion_temperature
    temperature_term = (1 + temperature / kelvin) ** (-1)
    pressure_term = (
        hydrostatic_pressure / non_dimensional_params.atmospheric_pressure_scaled
    )
    laplace_term = non_dimensional_params.laplace_pressure_scale
    depth_term = (
        -non_dimensional_params.hydrostatic_pressure_scale * height * mushy_layer_depth
    )

    return temperature_term * (1 + pressure_term + laplace_term + depth_term)


def calculate_gas_fraction(
    frozen_gas_fraction,
    solid_fraction,
    temperature,
    dissolved_gas_concentration,
    gas_density,
    non_dimensional_params: NonDimensionalParams,
):
    expansion_coefficient = non_dimensional_params.expansion_coefficient
    far_dissolved_gas_concentration = (
        non_dimensional_params.far_dissolved_concentration_scaled
    )
    concentration_ratio = non_dimensional_params.concentration_ratio
    buoyancy = non_dimensional_params.stokes_rise_velocity_scaled
    bubble_scale = non_dimensional_params.bubble_radius_scaled

    drag = calculate_drag(
        calculate_bubble_radius(solid_fraction, non_dimensional_params)
    )

    numerator = (
        expansion_coefficient
        * (1 - frozen_gas_fraction)
        * (
            far_dissolved_gas_concentration
            - dissolved_gas_concentration
            * (concentration_ratio / (concentration_ratio - temperature))
        )
    )
    interstitial_gas_velocity = drag * buoyancy * bubble_scale**2
    denominator = gas_density * (1 + interstitial_gas_velocity)

    return numerator / denominator
