# need to find a way to calculate gas_fraction and gas_density from state variables
import numpy as np
from .model.full_nonlinear_gas_fraction_solve import (
    calculate_gas_density as calculate_full_gas_density,
)


def get_boundary_conditions_full(
    non_dimensional_params, bottom_variables, top_variables
):
    chi = non_dimensional_params.expansion_coefficient
    far_dissolved_concentration_scaled = (
        non_dimensional_params.far_dissolved_concentration_scaled
    )
    (
        bottom_temperature,
        bottom_temperature_derivative,
        bottom_dissolved_gas,
        bottom_hydrostatic_pressure,
        bottom_frozen_gas_fraction,
        bottom_mushy_layer_depth,
    ) = bottom_variables
    (
        top_temperature,
        top_temperature_derivative,
        top_dissolved_gas,
        top_hydrostatic_pressure,
        top_frozen_gas_fraction,
        top_mushy_layer_depth,
    ) = top_variables
    top_gas_density = calculate_full_gas_density(
        0,
        top_mushy_layer_depth,
        top_temperature,
        top_hydrostatic_pressure,
        non_dimensional_params,
    )
    top_frozen_gas = (
        1 + (top_gas_density / (chi * far_dissolved_concentration_scaled))
    ) ** (-1)

    return np.array(
        [
            top_hydrostatic_pressure,
            top_temperature + 1,
            top_frozen_gas_fraction - top_frozen_gas,
            bottom_temperature,
            bottom_dissolved_gas - far_dissolved_concentration_scaled,
            bottom_temperature_derivative
            + bottom_mushy_layer_depth
            * non_dimensional_params.far_temperature_scaled
            * (1 - bottom_frozen_gas_fraction),
        ]
    )
