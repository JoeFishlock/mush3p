"""
The equations for solving the ideal model

All quantities are calculated from the smaller set of variables:
temperature
temperature_derivative
dissolved_gas_concentration
hydrostatic_pressure
frozen_gas_fraction
mushy_layer_depth

height (vertical coordinate)
"""

from typing import Union
from numpy.typing import NDArray
from .full import FullModel

Array = Union[NDArray, float]


class IdealModel(FullModel):
    """Class containing equations for thermally ideal model.
    The heat equation is unaware of the presence of a gas phase,
    as all phases have the same thermal properties."""

    def calculate_temperature_second_derivative(
        self,
        temperature_derivative: Array,
        gas_fraction: Array,
        frozen_gas_fraction: Array,
        mushy_layer_depth: Array,
        solid_fraction_derivative: Array,
        gas_fraction_derivative: Array,
    ) -> Array:
        stefan_number = self.params.stefan_number

        heating = (
            mushy_layer_depth
            * (1 + gas_fraction - frozen_gas_fraction)
            * temperature_derivative
            - mushy_layer_depth * stefan_number * solid_fraction_derivative
        )

        return heating
