"""
model.py contains the equations for solving the full model

All quantities are calculated from the smaller set of variables:
temperature
temperature_derivative
dissolved_gas_concentration
hydrostatic_pressure
frozen_gas_fraction
mushy_layer_depth

height (vertical coordinate)
"""

from typing import Union, Any
import numpy as np
from scipy.optimize import fsolve  # type: ignore
from numpy.typing import NDArray
from agate.params import FullNonDimensionalParams

#######################################################################
#                       mushy layer quantities                        #
#######################################################################

PORE_THROAT_SCALING: float = 0.5

Array = Union[NDArray, float]


def calculate_solid_salinity_in_mushy_layer(
    temperature: Array, params: FullNonDimensionalParams
) -> Array:
    return np.full_like(temperature, -params.concentration_ratio)


def calculate_liquid_salinity_in_mushy_layer(temperature: Array) -> Array:
    return -temperature


def calculate_liquid_darcy_velocity_in_mushy_layer(
    gas_fraction: Array, frozen_gas_fraction: Array
) -> Array:
    return gas_fraction - frozen_gas_fraction


def calculate_solid_fraction_in_mushy_layer(
    params: FullNonDimensionalParams, temperature: Array, frozen_gas_fraction: Array
) -> Array:
    concentration_ratio = params.concentration_ratio
    return (
        -(1 - frozen_gas_fraction) * temperature / (concentration_ratio - temperature)
    )


def calculate_liquid_fraction_in_mushy_layer(
    solid_fraction: Array, gas_fraction: Array
) -> Array:
    return 1 - solid_fraction - gas_fraction


def calculate_liquid_saturation_in_mushy_layer(
    solid_fraction: Array, liquid_fraction: Array
) -> Array:
    return liquid_fraction / (1 - solid_fraction)


def calculate_bubble_radius_in_mushy_layer(
    params: FullNonDimensionalParams, liquid_fraction: Array
) -> Array:
    return params.bubble_radius_scaled / (liquid_fraction**PORE_THROAT_SCALING)


def calculate_lag_in_mushy_layer(bubble_radius: Array) -> Array:
    lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
    lag = np.where(bubble_radius > 1, 0.5, lag)
    return lag


def calculate_drag_in_mushy_layer(bubble_radius: Array) -> Array:
    drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** 4)
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_darcy_velocity_in_mushy_layer(
    params: FullNonDimensionalParams,
    gas_fraction: Array,
    liquid_fraction: Array,
    liquid_darcy_velocity: Array,
) -> Array:
    bubble_radius = calculate_bubble_radius_in_mushy_layer(
        params=params, liquid_fraction=liquid_fraction
    )
    drag = calculate_drag_in_mushy_layer(bubble_radius=bubble_radius)
    lag = calculate_lag_in_mushy_layer(bubble_radius=bubble_radius)

    buoyancy_term = params.stokes_rise_velocity_scaled * drag
    liquid_term = 2 * lag * liquid_darcy_velocity / liquid_fraction

    return gas_fraction * (buoyancy_term + liquid_term)


def calculate_gas_density_in_mushy_layer(
    params: FullNonDimensionalParams,
    temperature: Array,
    hydrostatic_pressure: Array,
    height: Array,
) -> Array:
    temperature_term = (1 + temperature / params.kelvin_conversion_temperature) ** (-1)
    pressure_term = hydrostatic_pressure / params.atmospheric_pressure_scaled
    laplace_term = params.laplace_pressure_scale / params.atmospheric_pressure_scaled
    depth_term = (
        -params.hydrostatic_pressure_scale * height / params.atmospheric_pressure_scaled
    )

    return temperature_term * (1 + pressure_term + laplace_term + depth_term)


def calculate_gas_fraction_in_mushy_layer(
    params: FullNonDimensionalParams,
    solid_fraction,
    frozen_gas_fraction: Array,
    gas_density: Array,
    dissolved_gas_concentration: Array,
) -> Any:
    expansion_coefficient = params.expansion_coefficient
    far_dissolved_gas_concentration = params.far_dissolved_concentration_scaled

    def residual(gas_fraction: Array) -> Array:
        liquid_darcy_velocity = calculate_liquid_darcy_velocity_in_mushy_layer(
            gas_fraction=gas_fraction, frozen_gas_fraction=frozen_gas_fraction
        )

        liquid_fraction = calculate_liquid_fraction_in_mushy_layer(
            solid_fraction=solid_fraction, gas_fraction=gas_fraction
        )

        gas_darcy_velocity = calculate_gas_darcy_velocity_in_mushy_layer(
            params=params,
            gas_fraction=gas_fraction,
            liquid_fraction=liquid_fraction,
            liquid_darcy_velocity=liquid_darcy_velocity,
        )

        return (
            gas_density * (gas_fraction + gas_darcy_velocity) / expansion_coefficient
            + dissolved_gas_concentration * (liquid_fraction + liquid_darcy_velocity)
            - far_dissolved_gas_concentration * (1 - frozen_gas_fraction)
        )

    # TODO: make initial guess a numerical parameter <14-12-22, Joe Fishlock> #

    # TODO: write a unit test to test gas_fraction is 0 when no dissolved gas present
    # <14-12-22, Joe Fishlock> #

    initial_guess = np.full_like(solid_fraction, 0.01)
    return fsolve(residual, initial_guess)


def calculate_solid_fraction_derivative_in_mushy_layer(
    params: FullNonDimensionalParams,
    temperature: Array,
    temperature_derivative: Array,
    frozen_gas_fraction: Array,
) -> Array:
    concentration_ratio = params.concentration_ratio
    return (
        concentration_ratio
        * (1 - frozen_gas_fraction)
        * temperature_derivative
        / (concentration_ratio - temperature) ** 2
    )


def calculate_frozen_gas_at_top(
    params: FullNonDimensionalParams, gas_density_at_top: float
) -> float:
    expansion_coefficient = params.expansion_coefficient
    far_dissolved_concentration_scaled = params.far_dissolved_concentration_scaled
    return (
        1
        + gas_density_at_top
        / (expansion_coefficient * far_dissolved_concentration_scaled)
    ) ** (-1)
