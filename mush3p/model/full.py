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
from ..static_settings import (
    VOLUME_SUM_TOLERANCE,
)
from .full_nonlinear_gas_fraction_solve import (
    calculate_gas_fraction,
    calculate_gas_darcy_velocity,
    calculate_liquid_fraction,
    calculate_liquid_darcy_velocity,
    calculate_gas_density,
)


Array = Union[NDArray, float]


class FullModel:
    """Class containing full equations for system"""

    def __init__(
        self,
        params,
        height,
        temperature,
        temperature_derivative,
        dissolved_gas_concentration,
        hydrostatic_pressure,
        frozen_gas_fraction,
        mushy_layer_depth,
    ) -> None:
        self.params = params
        self.height = height
        self.temperature = temperature
        self.temperature_derivative = temperature_derivative
        self.dissolved_gas_concentration = dissolved_gas_concentration
        self.hydrostatic_pressure = hydrostatic_pressure
        self.frozen_gas_fraction = frozen_gas_fraction
        self.mushy_layer_depth = mushy_layer_depth

    @property
    def solid_salinity(self) -> Array:
        return np.full_like(self.temperature, -self.params.concentration_ratio)

    @property
    def liquid_salinity(self) -> Array:
        return -self.temperature

    @property
    def solid_fraction(self) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -(1 - self.frozen_gas_fraction)
            * self.temperature
            / (concentration_ratio - self.temperature)
        )

    @property
    def liquid_fraction(self) -> Array:
        return calculate_liquid_fraction(self.gas_fraction, self.solid_fraction)

    @property
    def gas_darcy_velocity(
        self,
    ) -> Array:
        return calculate_gas_darcy_velocity(
            self.solid_fraction,
            self.gas_fraction,
            self.params,
        )

    @property
    def gas_density(
        self,
    ):
        return calculate_gas_density(
            self.height,
            self.mushy_layer_depth,
            self.temperature,
            self.hydrostatic_pressure,
            self.params,
        )

    @property
    def gas_fraction(
        self,
    ) -> Any:
        return calculate_gas_fraction(
            self.frozen_gas_fraction,
            self.solid_fraction,
            self.temperature,
            self.dissolved_gas_concentration,
            self.gas_density,
            self.params,
        )

    @property
    def liquid_darcy_velocity(self):
        return calculate_liquid_darcy_velocity(
            self.gas_fraction, self.frozen_gas_fraction
        )

    @property
    def permeability(self) -> Array:
        liquid_permeability_reciprocal = (
            1 - self.liquid_fraction
        ) ** 2 / self.liquid_fraction**3
        reference = self.params.hele_shaw_permeability_scaled
        return ((1 / reference) + liquid_permeability_reciprocal) ** (-1)

    @property
    def saturation_concentration(self) -> Array:
        return np.full_like(self.temperature, 1)

    @property
    def nucleation_rate(self) -> Array:
        indicator = np.where(
            self.dissolved_gas_concentration >= self.saturation_concentration, 1, 0
        )

        return (
            self.liquid_fraction
            * indicator
            * (self.dissolved_gas_concentration - self.saturation_concentration)
        )

    @property
    def solid_fraction_derivative(
        self,
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -concentration_ratio
            * (1 - self.frozen_gas_fraction)
            * self.temperature_derivative
            / (concentration_ratio - self.temperature) ** 2
        )

    @property
    def gas_fraction_derivative(self) -> Array:
        """Numerically approximate the derivative with finite difference."""
        return np.gradient(self.gas_fraction, self.height)

    @property
    def hydrostatic_pressure_derivative(
        self,
    ) -> Array:
        return -self.mushy_layer_depth * self.liquid_darcy_velocity / self.permeability

    @property
    def effective_heat_capacity(self):
        solid_specific_heat_capacity_ratio = (
            self.params.solid_specific_heat_capacity_ratio
        )
        gas_specific_heat_capacity_ratio = self.params.gas_specific_heat_capacity_ratio
        density_ratio = self.params.gas_density_ratio
        return (
            1
            - (1 - solid_specific_heat_capacity_ratio) * self.solid_fraction
            - (1 - gas_specific_heat_capacity_ratio * density_ratio * self.gas_density)
            * self.gas_fraction
        )

    @property
    def effective_thermal_conductivity(self):
        gas_conductivity_ratio = self.params.gas_conductivity_ratio
        solid_conductivity_ratio = self.params.solid_conductivity_ratio
        return (
            1
            - (1 - solid_conductivity_ratio) * self.solid_fraction
            - (1 - gas_conductivity_ratio) * self.gas_fraction
        )

    @property
    def temperature_second_derivative(
        self,
    ) -> Array:
        stefan_number = self.params.stefan_number
        gas_specific_heat_capacity_ratio = self.params.gas_specific_heat_capacity_ratio
        density_ratio = self.params.gas_density_ratio
        gas_conductivity_ratio = self.params.gas_conductivity_ratio
        solid_conductivity_ratio = self.params.solid_conductivity_ratio

        heat_capacity_term = (
            self.mushy_layer_depth
            * self.effective_heat_capacity
            * self.temperature_derivative
        )
        liquid_advection_term = (
            self.mushy_layer_depth
            * self.liquid_darcy_velocity
            * self.temperature_derivative
        )
        gas_advection_term = (
            self.mushy_layer_depth
            * density_ratio
            * self.gas_density
            * gas_specific_heat_capacity_ratio
            * self.gas_darcy_velocity
            * self.temperature_derivative
        )
        latent_heat_term = (
            -self.mushy_layer_depth * stefan_number * self.solid_fraction_derivative
        )
        conductivity_change_term = (
            (1 - solid_conductivity_ratio) * self.solid_fraction_derivative
            + (1 - gas_conductivity_ratio) * self.gas_fraction_derivative
        ) * self.temperature_derivative

        return (1 / self.effective_thermal_conductivity) * (
            heat_capacity_term
            + liquid_advection_term
            + gas_advection_term
            + latent_heat_term
            + conductivity_change_term
        )

    @property
    def dissolved_gas_concentration_derivative(
        self,
    ) -> Array:

        damkholer_number = self.params.damkholer_number
        freezing = self.dissolved_gas_concentration * self.solid_fraction_derivative
        dissolution = -damkholer_number * self.mushy_layer_depth * self.nucleation_rate

        return (freezing + dissolution) / (
            1 - self.frozen_gas_fraction - self.solid_fraction
        )

    @property
    def check_volume_fractions_sum_to_one(self):
        if (
            np.max(
                np.abs(
                    self.solid_fraction + self.liquid_fraction + self.gas_fraction - 1
                )
            )
            > VOLUME_SUM_TOLERANCE
        ):
            return False
        return True

    @property
    def ode_fun(self):

        if not self.check_volume_fractions_sum_to_one:
            raise ValueError("Volume fractions do not sum to 1")

        return np.vstack(
            (
                self.temperature_derivative,
                self.temperature_second_derivative,
                self.dissolved_gas_concentration_derivative,
                self.hydrostatic_pressure_derivative,
                np.zeros_like(self.temperature),
                np.zeros_like(self.temperature),
            )
        )
