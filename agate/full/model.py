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
    mushy_layer_depth: Array,
    height: Array,
) -> Array:
    temperature_term = (1 + temperature / params.kelvin_conversion_temperature) ** (-1)
    pressure_term = hydrostatic_pressure / params.atmospheric_pressure_scaled
    laplace_term = params.laplace_pressure_scale / params.atmospheric_pressure_scaled
    depth_term = (
        -params.hydrostatic_pressure_scale
        * height
        * mushy_layer_depth
        / params.atmospheric_pressure_scaled
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


def calculate_permeability_in_mushy_layer(
    params: FullNonDimensionalParams, liquid_fraction: Array
) -> Array:
    liquid_permeability_reciprocal = (1 - liquid_fraction) ** 2 / liquid_fraction**3
    return (
        1 / params.hele_shaw_permeability_scaled + liquid_permeability_reciprocal
    ) ** (-1)


def calculate_saturation_concentration_in_mushy_layer(temperature: Array) -> Array:
    return np.full_like(temperature, 1)


def calculate_unconstrained_nucleation_rate_in_mushy_layer(
    dissolved_gas_concentration: Array, saturation_concentration: Array
) -> Array:
    return dissolved_gas_concentration - saturation_concentration


def calculate_nucleation_indicator_in_mushy_layer(
    dissolved_gas_concentration: Array, liquid_saturation: Array
) -> Array:
    indicator = np.where(
        (liquid_saturation == 1) and (dissolved_gas_concentration >= 1), 1, 0
    )
    indicator = np.where(
        (0 < liquid_saturation) and (liquid_saturation < 1), 1, indicator
    )
    return indicator


def calculate_nucleation_rate_in_mushy_layer(
    temperature: Array, dissolved_gas_concentration: Array, liquid_saturation: Array
) -> Array:
    saturation_concentration = calculate_saturation_concentration_in_mushy_layer(
        temperature=temperature
    )
    unconstrained_nucleation_rate = (
        calculate_unconstrained_nucleation_rate_in_mushy_layer(
            dissolved_gas_concentration=dissolved_gas_concentration,
            saturation_concentration=saturation_concentration,
        )
    )
    nucleation_indicator = calculate_nucleation_indicator_in_mushy_layer(
        dissolved_gas_concentration=dissolved_gas_concentration,
        liquid_saturation=liquid_saturation,
    )

    return nucleation_indicator * unconstrained_nucleation_rate


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


def calculate_gas_fraction_derivative_in_mushy_layer(
    gas_fraction: Array, height: Array
) -> Array:
    """Numerically approximate the derivative with finite difference."""
    return np.gradient(gas_fraction, height)


def calculate_hydrostatic_pressure_derivative_in_mushy_layer(
    permeability: Array, liquid_darcy_velocity: Array, mushy_layer_depth: Array
) -> Array:
    return -mushy_layer_depth * liquid_darcy_velocity / permeability


def calculate_temperature_derivative_in_mushy_layer(
    temperature_derivative: Array,
) -> Array:
    return temperature_derivative


def calculate_temperature_second_derivative_in_mushy_layer(
    params: FullNonDimensionalParams,
    temperature_derivative: Array,
    gas_fraction: Array,
    frozen_gas_fraction: Array,
    mushy_layer_depth: Array,
    solid_fraction_derivative: Array,
    gas_fraction_derivative: Array,
) -> Array:
    stefan_number = params.stefan_number
    gas_conductivity_ratio = params.gas_conductivity_ratio

    heating = (
        mushy_layer_depth * (1 - frozen_gas_fraction) * temperature_derivative
        - mushy_layer_depth * stefan_number * solid_fraction_derivative
    )

    gas_insulation = (
        (1 - gas_conductivity_ratio) * gas_fraction_derivative * temperature_derivative
    )

    return (heating + gas_insulation) / (
        1 - (1 - gas_conductivity_ratio) * gas_fraction
    )


def calculate_dissolved_gas_concentration_derivative_in_mushy_layer(
    params: FullNonDimensionalParams,
    dissolved_gas_concentration: Array,
    solid_fraction_derivative: Array,
    frozen_gas_fraction: Array,
    solid_fraction: Array,
    mushy_layer_depth,
    nucleation_rate: Array,
) -> Array:

    damkholer_number = params.damkholer_number
    freezing = dissolved_gas_concentration * solid_fraction_derivative
    dissolution = -damkholer_number * mushy_layer_depth * nucleation_rate

    return (freezing + dissolution) / (1 - frozen_gas_fraction - solid_fraction)


def calculate_frozen_gas_fraction_derivative_in_mushy_layer(
    temperature: Array,
) -> Array:
    return np.zeros_like(temperature)


def calculate_mushy_layer_depth_derivative_in_mushy_layer(temperature: Array) -> Array:
    return np.zeros_like(temperature)


def calculate_frozen_gas_at_top_in_mushy_layer(
    params: FullNonDimensionalParams, gas_density_at_top: float
) -> float:
    expansion_coefficient = params.expansion_coefficient
    far_dissolved_concentration_scaled = params.far_dissolved_concentration_scaled
    return (
        1
        + gas_density_at_top
        / (expansion_coefficient * far_dissolved_concentration_scaled)
    ) ** (-1)


def ode_fun_in_mushy_layer(
    params: FullNonDimensionalParams, height: Array, variables: Any
) -> Any:
    (
        temperature,
        temperature_derivative,
        dissolved_gas_concentration,
        hydrostatic_pressure,
        frozen_gas_fraction,
        mushy_layer_depth,
    ) = variables

    solid_fraction = calculate_solid_fraction_in_mushy_layer(
        params=params, temperature=temperature, frozen_gas_fraction=frozen_gas_fraction
    )

    solid_fraction_derivative = calculate_solid_fraction_derivative_in_mushy_layer(
        params=params,
        temperature=temperature,
        temperature_derivative=temperature_derivative,
        frozen_gas_fraction=frozen_gas_fraction,
    )

    gas_density = calculate_gas_density_in_mushy_layer(
        params=params,
        temperature=temperature,
        hydrostatic_pressure=hydrostatic_pressure,
        mushy_layer_depth=mushy_layer_depth,
        height=height,
    )

    gas_fraction = calculate_gas_fraction_in_mushy_layer(
        params=params,
        solid_fraction=solid_fraction,
        frozen_gas_fraction=frozen_gas_fraction,
        gas_density=gas_density,
        dissolved_gas_concentration=dissolved_gas_concentration,
    )

    gas_fraction_derivative = calculate_gas_fraction_derivative_in_mushy_layer(
        gas_fraction=gas_fraction, height=height
    )

    liquid_fraction = calculate_liquid_fraction_in_mushy_layer(
        solid_fraction=solid_fraction, gas_fraction=gas_fraction
    )

    liquid_saturation = calculate_liquid_saturation_in_mushy_layer(
        solid_fraction=solid_fraction, liquid_fraction=liquid_fraction
    )

    nucleation_rate = calculate_nucleation_rate_in_mushy_layer(
        temperature=temperature,
        dissolved_gas_concentration=dissolved_gas_concentration,
        liquid_saturation=liquid_saturation,
    )

    permeability = calculate_permeability_in_mushy_layer(
        params=params, liquid_fraction=liquid_fraction
    )

    liquid_darcy_velocity = calculate_liquid_darcy_velocity_in_mushy_layer(
        gas_fraction=gas_fraction, frozen_gas_fraction=frozen_gas_fraction
    )

    return np.vstack(
        (
            calculate_temperature_derivative_in_mushy_layer(
                temperature_derivative=temperature_derivative
            ),
            calculate_temperature_second_derivative_in_mushy_layer(
                params=params,
                temperature_derivative=temperature_derivative,
                gas_fraction=gas_fraction,
                frozen_gas_fraction=frozen_gas_fraction,
                mushy_layer_depth=mushy_layer_depth,
                solid_fraction_derivative=solid_fraction_derivative,
                gas_fraction_derivative=gas_fraction_derivative,
            ),
            calculate_dissolved_gas_concentration_derivative_in_mushy_layer(
                params=params,
                dissolved_gas_concentration=dissolved_gas_concentration,
                solid_fraction_derivative=solid_fraction_derivative,
                frozen_gas_fraction=frozen_gas_fraction,
                solid_fraction=solid_fraction,
                mushy_layer_depth=mushy_layer_depth,
                nucleation_rate=nucleation_rate,
            ),
            calculate_hydrostatic_pressure_derivative_in_mushy_layer(
                permeability=permeability,
                liquid_darcy_velocity=liquid_darcy_velocity,
                mushy_layer_depth=mushy_layer_depth,
            ),
            calculate_frozen_gas_fraction_derivative_in_mushy_layer(
                temperature=temperature
            ),
            calculate_mushy_layer_depth_derivative_in_mushy_layer(
                temperature=temperature
            ),
        )
    )
