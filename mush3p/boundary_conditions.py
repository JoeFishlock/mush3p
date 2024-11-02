import numpy as np
from numpy.typing import NDArray
from .model.full_nonlinear_gas_fraction_solve import (
    calculate_gas_density as calculate_full_gas_density,
)
from .params import NonDimensionalParams


def get_boundary_conditions(
    non_dimensional_params: NonDimensionalParams,
    bottom_variables: NDArray,
    top_variables: NDArray,
) -> NDArray:
    """Function to return the boundary conditions for scipy solve_BVP.

    The returned array is zero when the boundary conditions are satisfied.

    Args:
        non_dimensional_params (NonDimensionalParams): Non-dimensional parameters
        bottom_variables (NDArray): Array of the solution variables evaluated at the bottom boundary.
        top_variables (NDArray): Array of the solution variables evaluated at the top boundary.

    Returns:
        NDArray: residual of the boundary conditions
    """

    OPTIONS = {
        "full": BoundaryConditionsFull,
        "incompressible": BoundaryConditionsIncompressible,
        "reduced": BoundaryConditionsReduced,
        "instant": BoundaryConditionsInstant,
    }
    return OPTIONS[non_dimensional_params.model_choice](
        non_dimensional_params, bottom_variables, top_variables
    ).boundary_conditions


# The boundary conditions for each model are implemented as a class which must implement
# the boundary_conditions method
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


class BoundaryConditionsInstant(BoundaryConditionsReduced):
    def __init__(self, non_dimensional_params, bottom_variables, top_variables):
        self.non_dimensional_params = non_dimensional_params
        (
            self.bottom_temperature,
            self.bottom_temperature_derivative,
            self.bottom_hydrostatic_pressure,
            self.bottom_frozen_gas_fraction,
            self.bottom_mushy_layer_depth,
        ) = bottom_variables
        (
            self.top_temperature,
            self.top_temperature_derivative,
            self.top_hydrostatic_pressure,
            self.top_frozen_gas_fraction,
            self.top_mushy_layer_depth,
        ) = top_variables

    @property
    def boundary_conditions(self):

        return np.array(
            [
                self.top_hydrostatic_pressure,
                self.top_temperature + 1,
                self.top_frozen_gas_fraction - self.top_frozen_gas,
                self.bottom_temperature,
                self.bottom_temperature_derivative
                + self.bottom_mushy_layer_depth
                * self.non_dimensional_params.far_temperature_scaled,
            ]
        )
