"""
The equations for solving the incompressible model

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
from .full import FullModel

Array = Union[NDArray, float]


class IncompressibleModel(FullModel):
    """Class containing equations with no gas compressibility.
    The non dimensional gas density is set to 1.0"""

    def calculate_gas_density(
        self,
        temperature: Array,
    ) -> Array:
        return np.ones_like(temperature)

    def ode_fun(self, height: Array, variables: Any) -> Any:
        (
            temperature,
            temperature_derivative,
            dissolved_gas_concentration,
            hydrostatic_pressure,
            frozen_gas_fraction,
            mushy_layer_depth,
        ) = variables

        solid_fraction = self.calculate_solid_fraction(
            temperature=temperature, frozen_gas_fraction=frozen_gas_fraction
        )

        solid_fraction_derivative = self.calculate_solid_fraction_derivative(
            temperature=temperature,
            temperature_derivative=temperature_derivative,
            frozen_gas_fraction=frozen_gas_fraction,
        )

        gas_density = self.calculate_gas_density(
            temperature=temperature,
        )

        gas_fraction = self.calculate_gas_fraction(
            solid_fraction=solid_fraction,
            frozen_gas_fraction=frozen_gas_fraction,
            gas_density=gas_density,
            dissolved_gas_concentration=dissolved_gas_concentration,
        )

        gas_fraction_derivative = self.calculate_gas_fraction_derivative(
            gas_fraction=gas_fraction, height=height
        )

        liquid_fraction = self.calculate_liquid_fraction(
            solid_fraction=solid_fraction, gas_fraction=gas_fraction
        )

        nucleation_rate = self.calculate_nucleation_rate(
            temperature=temperature,
            dissolved_gas_concentration=dissolved_gas_concentration,
        )

        permeability = self.calculate_permeability(liquid_fraction=liquid_fraction)

        liquid_darcy_velocity = self.calculate_liquid_darcy_velocity(
            gas_fraction=gas_fraction, frozen_gas_fraction=frozen_gas_fraction
        )

        if not self.check_volume_fractions_sum_to_one(
            solid_fraction, liquid_fraction, gas_fraction
        ):
            raise ValueError("Volume fractions do not sum to 1")

        return np.vstack(
            (
                self.calculate_temperature_derivative(
                    temperature_derivative=temperature_derivative
                ),
                self.calculate_temperature_second_derivative(
                    temperature_derivative=temperature_derivative,
                    gas_fraction=gas_fraction,
                    frozen_gas_fraction=frozen_gas_fraction,
                    mushy_layer_depth=mushy_layer_depth,
                    solid_fraction_derivative=solid_fraction_derivative,
                    gas_fraction_derivative=gas_fraction_derivative,
                ),
                self.calculate_dissolved_gas_concentration_derivative(
                    dissolved_gas_concentration=dissolved_gas_concentration,
                    solid_fraction_derivative=solid_fraction_derivative,
                    frozen_gas_fraction=frozen_gas_fraction,
                    solid_fraction=solid_fraction,
                    mushy_layer_depth=mushy_layer_depth,
                    nucleation_rate=nucleation_rate,
                ),
                self.calculate_hydrostatic_pressure_derivative(
                    permeability=permeability,
                    liquid_darcy_velocity=liquid_darcy_velocity,
                    mushy_layer_depth=mushy_layer_depth,
                ),
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
            _,
            hydrostatic_pressure_at_top,
            frozen_gas_fraction_at_top,
            mushy_layer_depth_at_top,
        ) = variables_at_top
        (
            temperature_at_bottom,
            temperature_derivative_at_bottom,
            dissolved_gas_concentration_at_bottom,
            _,
            frozen_gas_fraction_at_bottom,
            mushy_layer_depth_at_bottom,
        ) = variables_at_bottom

        gas_density_at_top = self.calculate_gas_density(
            temperature=temperature_at_top,
        )

        return np.array(
            [
                hydrostatic_pressure_at_top,
                temperature_at_top + 1,
                frozen_gas_fraction_at_top
                - self.calculate_frozen_gas_at_top(
                    gas_density_at_top=gas_density_at_top
                ),
                temperature_at_bottom,
                dissolved_gas_concentration_at_bottom
                - self.params.far_dissolved_concentration_scaled,
                temperature_derivative_at_bottom
                + mushy_layer_depth_at_bottom
                * self.params.far_temperature_scaled
                * (1 - frozen_gas_fraction_at_bottom),
            ]
        )

    def calculate_all_variables(
        self,
        temperature,
        temperature_derivative,
        dissolved_gas_concentration,
        hydrostatic_pressure,
        frozen_gas_fraction,
        mushy_layer_depth,
        height,
    ):
        solid_salinity = self.calculate_solid_salinity(temperature)
        liquid_salinity = self.calculate_liquid_salinity(temperature)
        solid_fraction = self.calculate_solid_fraction(temperature, frozen_gas_fraction)
        gas_density = self.calculate_gas_density(temperature)
        gas_fraction = self.calculate_gas_fraction(
            solid_fraction,
            frozen_gas_fraction,
            gas_density,
            dissolved_gas_concentration,
        )
        liquid_fraction = self.calculate_liquid_fraction(solid_fraction, gas_fraction)
        liquid_darcy_velocity = self.calculate_liquid_darcy_velocity(
            gas_fraction, frozen_gas_fraction
        )
        gas_darcy_velocity = self.calculate_gas_darcy_velocity(
            gas_fraction, liquid_fraction, liquid_darcy_velocity
        )
        return (
            solid_salinity,
            liquid_salinity,
            solid_fraction,
            liquid_fraction,
            gas_fraction,
            gas_density,
            liquid_darcy_velocity,
            gas_darcy_velocity,
        )
