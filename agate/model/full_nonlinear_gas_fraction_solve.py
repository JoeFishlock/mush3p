import numpy as np
from scipy.optimize import fsolve
from ..static_settings import GAS_FRACTION_GUESS


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
    Hartholt_drag = lambda L: (1 - 1.5 * L + 1.5 * L**5 - L**6) / (1 + 1.5 * L**5)
    drag = np.where(bubble_radius < 0, 1, Hartholt_drag(bubble_radius))
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_darcy_velocity(solid_fraction, gas_fraction, non_dimensional_params):
    exponent = non_dimensional_params.pore_throat_exponent
    bubble_radius = calculate_bubble_radius(solid_fraction, non_dimensional_params)
    drag = calculate_drag(bubble_radius)

    return (
        gas_fraction
        * non_dimensional_params.stokes_rise_velocity_scaled
        * drag
        * (bubble_radius**2)
        * ((1 - solid_fraction) ** (2 * exponent))
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
            solid_fraction, gas_fraction, non_dimensional_params
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
