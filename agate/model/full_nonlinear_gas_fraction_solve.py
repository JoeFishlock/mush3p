import numpy as np
from scipy.optimize import fsolve
from ..static_settings import GAS_FRACTION_GUESS, PORE_THROAT_SCALING, DRAG_EXPONENT


def calculate_liquid_darcy_velocity(gas_fraction, frozen_gas_fraction):
    return gas_fraction - frozen_gas_fraction


def calculate_liquid_fraction(gas_fraction, solid_fraction):
    return 1 - solid_fraction - gas_fraction


def calculate_bubble_radius(liquid_fraction, non_dimensional_params):
    exponent = PORE_THROAT_SCALING
    return non_dimensional_params.bubble_radius_scaled / (liquid_fraction**exponent)


def calculate_lag(bubble_radius):
    lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
    lag = np.where(bubble_radius > 1, 0.5, lag)
    return lag


def calculate_drag(bubble_radius):
    exponent = DRAG_EXPONENT
    drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** exponent)
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_darcy_velocity(
    liquid_fraction, gas_fraction, frozen_gas_fraction, non_dimensional_params
):

    bubble_radius = calculate_bubble_radius(liquid_fraction, non_dimensional_params)
    liquid_darcy_velocity = calculate_liquid_darcy_velocity(
        gas_fraction, frozen_gas_fraction
    )

    buoyancy_term = non_dimensional_params.stokes_rise_velocity_scaled * calculate_drag(
        bubble_radius
    )
    liquid_term = (
        2 * calculate_lag(bubble_radius) * liquid_darcy_velocity / liquid_fraction
    )

    return gas_fraction * (buoyancy_term + liquid_term)


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
    dissolved_gas_concentration,
    gas_density,
    non_dimensional_params,
):
    expansion_coefficient = non_dimensional_params.expansion_coefficient
    far_dissolved_gas_concentration = (
        non_dimensional_params.far_dissolved_concentration_scaled
    )

    def residual(gas_fraction):
        liquid_darcy_velocity = calculate_liquid_darcy_velocity(
            gas_fraction, frozen_gas_fraction
        )
        liquid_fraction = calculate_liquid_fraction(gas_fraction, solid_fraction)
        gas_darcy_velocity = calculate_gas_darcy_velocity(
            liquid_fraction, gas_fraction, frozen_gas_fraction, non_dimensional_params
        )

        gas_term = gas_density * (gas_fraction + gas_darcy_velocity)
        dissolved_gas_term = (
            expansion_coefficient
            * dissolved_gas_concentration
            * (liquid_fraction + liquid_darcy_velocity)
        )
        boundary_term = (
            expansion_coefficient
            * far_dissolved_gas_concentration
            * (1 - frozen_gas_fraction)
        )
        return gas_term + dissolved_gas_term - boundary_term

    # TODO: write a unit test to test gas_fraction is 0 when no dissolved gas present
    # <14-12-22, Joe Fishlock> #{data_path}/gas_fraction_model_error.pdf

    initial_guess = np.full_like(solid_fraction, GAS_FRACTION_GUESS)
    return fsolve(residual, initial_guess)
