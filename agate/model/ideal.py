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

from .full import FullModel


class IdealModel(FullModel):
    """Class containing equations for thermally ideal model.
    The heat equation is unaware of the presence of a gas phase,
    as all phases have the same thermal properties."""

    @property
    def calculate_temperature_second_derivative(
        self,
    ):
        stefan_number = self.params.stefan_number

        heating = (
            self.mushy_layer_depth
            * (1 + self.gas_fraction - self.frozen_gas_fraction)
            * self.temperature_derivative
            - self.mushy_layer_depth * stefan_number * self.solid_fraction_derivative
        )

        return heating
