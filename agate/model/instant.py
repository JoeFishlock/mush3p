"""
The equations for solving the full model

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
from numpy.typing import NDArray
from .reduced import ReducedModel

Array = Union[NDArray, float]


class InstantNucleationModel(ReducedModel):
    """Class containing full equations for system"""

    INITIAL_VARIABLES = np.vstack(
        (
            ReducedModel.INITIAL_TEMPERATURE,
            ReducedModel.INITIAL_TEMPERATURE_DERIVATIVE,
            ReducedModel.INITIAL_HYDROSTATIC_PRESSURE,
            ReducedModel.INITIAL_FROZEN_GAS_FRACTION,
            ReducedModel.INITIAL_MUSHY_LAYER_DEPTH,
        )
    )

    def __init__(self, params) -> None:
        self.params = params

    def calculate_dissolved_gas_concentration(self, liquid_fraction: Array):
        return np.minimum(
            np.ones_like(liquid_fraction),
            self.params.far_dissolved_concentration_scaled / liquid_fraction,
        )

    def ode_fun(self, height: Array, variables: Any) -> Any:
        (
            temperature,
            temperature_derivative,
            hydrostatic_pressure,
            frozen_gas_fraction,
            mushy_layer_depth,
        ) = variables

        solid_fraction = self.calculate_solid_fraction(temperature=temperature)

        solid_fraction_derivative = self.calculate_solid_fraction_derivative(
            temperature=temperature,
            temperature_derivative=temperature_derivative,
        )

        liquid_fraction = self.calculate_liquid_fraction(solid_fraction=solid_fraction)
        dissolved_gas_concentration = self.calculate_dissolved_gas_concentration(
            liquid_fraction=liquid_fraction
        )
        nucleation_rate = self.calculate_nucleation_rate(
            temperature=temperature,
            dissolved_gas_concentration=dissolved_gas_concentration,
        )

        if not self.check_volume_fractions_sum_to_one(solid_fraction, liquid_fraction):
            raise ValueError("Volume fractions do not sum to 1")

        return np.vstack(
            (
                self.calculate_temperature_derivative(
                    temperature_derivative=temperature_derivative
                ),
                self.calculate_temperature_second_derivative(
                    temperature_derivative=temperature_derivative,
                    mushy_layer_depth=mushy_layer_depth,
                    solid_fraction_derivative=solid_fraction_derivative,
                ),
                self.calculate_zero_derivative(temperature=temperature),
                self.calculate_zero_derivative(temperature=temperature),
                self.calculate_zero_derivative(temperature=temperature),
            )
        )

    def boundary_conditions(
        self,
        variables_at_bottom: Any,
        variables_at_top: Any,
    ) -> Array:
        (
            temperature_at_top,
            _,
            hydrostatic_pressure_at_top,
            frozen_gas_fraction_at_top,
            _,
        ) = variables_at_top
        (
            temperature_at_bottom,
            temperature_derivative_at_bottom,
            _,
            _,
            mushy_layer_depth_at_bottom,
        ) = variables_at_bottom

        return np.array(
            [
                hydrostatic_pressure_at_top,
                temperature_at_top + 1,
                frozen_gas_fraction_at_top - self.calculate_frozen_gas_at_top(),
                temperature_at_bottom,
                temperature_derivative_at_bottom
                + mushy_layer_depth_at_bottom * self.params.far_temperature_scaled,
            ]
        )


# TODO: use a setter to check small variation in results <03-01-23, Joe Fishlock> #
# def check_variation_is_small(array):
#     max_difference = np.max(np.abs(np.diff(array)))
#     if max_difference > DIFFERENCE_TOLERANCE:
#         return False
#     return True
