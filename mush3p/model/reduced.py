from typing import Any
import numpy as np
from numpy.typing import NDArray

from .incompressible import IncompressibleModel
from ..static_settings import VOLUME_SUM_TOLERANCE
from .full_nonlinear_gas_fraction_solve import (
    calculate_gas_fraction,
)


class ReducedModel(IncompressibleModel):
    """Implement the equations for the reduced model

    The reduced model is an approximation to the full model where the exsolved gas
    volume fraction is small and incompressible.
    Under this approximation scheme the solid and liquid volume fractions sum to 1
    and the exsolution of gas does not drive a liquid flow.
    """

    @property
    def liquid_darcy_velocity(self) -> NDArray:
        return np.zeros_like(self.temperature)

    @property
    def solid_fraction(self) -> NDArray:
        concentration_ratio = self.params.concentration_ratio
        return self.temperature / (self.temperature - concentration_ratio)

    @property
    def liquid_fraction(self) -> NDArray:
        return 1 - self.solid_fraction

    @property
    def gas_fraction(
        self,
    ) -> Any:
        return calculate_gas_fraction(
            0,
            self.solid_fraction,
            self.temperature,
            self.dissolved_gas_concentration,
            1,
            self.params,
        )

    @property
    def solid_fraction_derivative(
        self,
    ) -> NDArray:
        concentration_ratio = self.params.concentration_ratio
        return (
            -concentration_ratio
            * self.temperature_derivative
            / ((self.temperature - concentration_ratio) ** 2)
        )

    @property
    def hydrostatic_pressure_derivative(
        self,
    ) -> NDArray:
        return np.zeros_like(self.temperature)

    @property
    def effective_heat_capacity(self):
        solid_specific_heat_capacity_ratio = (
            self.params.solid_specific_heat_capacity_ratio
        )
        return 1 - (1 - solid_specific_heat_capacity_ratio) * self.solid_fraction

    @property
    def effective_thermal_conductivity(self):
        solid_conductivity_ratio = self.params.solid_conductivity_ratio
        return 1 - (1 - solid_conductivity_ratio) * self.solid_fraction

    @property
    def temperature_second_derivative(
        self,
    ) -> NDArray:
        stefan_number = self.params.stefan_number
        solid_conductivity_ratio = self.params.solid_conductivity_ratio

        heat_capacity_term = (
            self.mushy_layer_depth
            * self.effective_heat_capacity
            * self.temperature_derivative
        )
        latent_heat_term = (
            -self.mushy_layer_depth * stefan_number * self.solid_fraction_derivative
        )
        conductivity_change_term = (
            (1 - solid_conductivity_ratio) * self.solid_fraction_derivative
        ) * self.temperature_derivative

        return (1 / self.effective_thermal_conductivity) * (
            heat_capacity_term + latent_heat_term + conductivity_change_term
        )

    @property
    def dissolved_gas_concentration_derivative(
        self,
    ) -> NDArray:

        damkholer_number = self.params.damkholer_number
        dissolution = -damkholer_number * self.mushy_layer_depth * self.nucleation_rate

        return (1 / self.liquid_fraction) * (
            self.dissolved_gas_concentration * self.solid_fraction_derivative
            + dissolution
        )

    @property
    def check_volume_fractions_sum_to_one(self):
        if (
            np.max(np.abs(self.solid_fraction + self.liquid_fraction - 1))
            > VOLUME_SUM_TOLERANCE
        ):
            return False
        return True
