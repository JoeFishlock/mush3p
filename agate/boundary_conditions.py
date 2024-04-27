# need to find a way to calculate gas_fraction and gas_density from state variables
import numpy as np
from .model.full_nonlinear_gas_fraction_solve import (
    calculate_gas_density as calculate_full_gas_density,
)


def get_boundary_conditions(non_dimensional_params, bottom_variables, top_variables):
    OPTIONS = {
        "full": BoundaryConditionsFull,
        "incompressible": BoundaryConditionsIncompressible,
        "ideal": BoundaryConditionsFull,
        "reduced": BoundaryConditionsReduced,
    }
    return OPTIONS[non_dimensional_params.model_choice](
        non_dimensional_params, bottom_variables, top_variables
    ).boundary_conditions


class BoundaryConditionsFull:
    def __init__(self, non_dimensional_params, bottom_variables, top_variables):
        self.non_dimensional_params = non_dimensional_params
        (
            self.bottom_temperature,
            self.bottom_temperature_derivative,
            self.bottom_dissolved_gas,
            self.bottom_hydrostatic_pressure,
            self.bottom_frozen_gas_fraction,
            self.bottom_mushy_layer_depth,
        ) = bottom_variables
        (
            self.top_temperature,
            self.top_temperature_derivative,
            self.top_dissolved_gas,
            self.top_hydrostatic_pressure,
            self.top_frozen_gas_fraction,
            self.top_mushy_layer_depth,
        ) = top_variables

    @property
    def top_gas_density(self):
        return calculate_full_gas_density(
            0,
            self.top_mushy_layer_depth,
            self.top_temperature,
            self.top_hydrostatic_pressure,
            self.non_dimensional_params,
        )

    @property
    def top_frozen_gas(self):
        chi = self.non_dimensional_params.expansion_coefficient
        far_dissolved_concentration_scaled = (
            self.non_dimensional_params.far_dissolved_concentration_scaled
        )
        return (
            1 + (self.top_gas_density / (chi * far_dissolved_concentration_scaled))
        ) ** (-1)

    @property
    def boundary_conditions(self):
        far_dissolved_concentration_scaled = (
            self.non_dimensional_params.far_dissolved_concentration_scaled
        )

        return np.array(
            [
                self.top_hydrostatic_pressure,
                self.top_temperature + 1,
                self.top_frozen_gas_fraction - self.top_frozen_gas,
                self.bottom_temperature,
                self.bottom_dissolved_gas - far_dissolved_concentration_scaled,
                self.bottom_temperature_derivative
                + self.bottom_mushy_layer_depth
                * self.non_dimensional_params.far_temperature_scaled
                * (1 - self.bottom_frozen_gas_fraction),
            ]
        )


class BoundaryConditionsIncompressible(BoundaryConditionsFull):
    @property
    def top_gas_density(self):
        return 1


class BoundaryConditionsReduced(BoundaryConditionsFull):
    @property
    def top_frozen_gas(self):
        chi = self.non_dimensional_params.expansion_coefficient
        far_dissolved_concentration_scaled = (
            self.non_dimensional_params.far_dissolved_concentration_scaled
        )
        return chi * far_dissolved_concentration_scaled

    @property
    def boundary_conditions(self):
        far_dissolved_concentration_scaled = (
            self.non_dimensional_params.far_dissolved_concentration_scaled
        )

        return np.array(
            [
                self.top_hydrostatic_pressure,
                self.top_temperature + 1,
                self.top_frozen_gas_fraction - self.top_frozen_gas,
                self.bottom_temperature,
                self.bottom_dissolved_gas - far_dissolved_concentration_scaled,
                self.bottom_temperature_derivative
                + self.bottom_mushy_layer_depth
                * self.non_dimensional_params.far_temperature_scaled,
            ]
        )
