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
from .full import FullModel

Array = Union[NDArray, float]


class InstantNucleationModel(FullModel):
    """Class containing full equations for system"""

    INITIAL_VARIABLES = np.vstack(
        (
            FullModel.INITIAL_TEMPERATURE,
            FullModel.INITIAL_TEMPERATURE_DERIVATIVE,
            FullModel.INITIAL_HYDROSTATIC_PRESSURE,
            FullModel.INITIAL_FROZEN_GAS_FRACTION,
            FullModel.INITIAL_MUSHY_LAYER_DEPTH,
        )
    )

    def __init__(self, params) -> None:
        self.params = params

    def calculate_dissolved_gas_concentration(self, liquid_fraction: Array):
        return np.minimum(
            np.ones_like(liquid_fraction),
            self.params.far_dissolved_concentration_scaled / liquid_fraction,
        )

    def calculate_liquid_darcy_velocity(self, temperature: Array) -> Array:
        return np.zeros_like(temperature)

    def calculate_solid_fraction(self, temperature: Array) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return temperature / (temperature - concentration_ratio)

    def calculate_liquid_fraction(self, solid_fraction: Array) -> Array:
        return 1 - solid_fraction

    def calculate_gas_darcy_velocity(
        self,
        gas_fraction: Array,
        liquid_fraction: Array,
    ) -> Array:
        bubble_radius = self.calculate_bubble_radius(liquid_fraction=liquid_fraction)
        drag = self.calculate_drag(bubble_radius=bubble_radius)

        buoyancy_term = self.params.stokes_rise_velocity_scaled * drag

        return gas_fraction * buoyancy_term

    def calculate_gas_density(
        self,
        temperature: Array,
    ) -> Array:
        return np.ones_like(temperature)

    def calculate_gas_fraction(
        self,
        solid_fraction,
        frozen_gas_fraction: Array,
        gas_density: Array,
        dissolved_gas_concentration: Array,
    ) -> Any:
        expansion_coefficient = self.params.expansion_coefficient
        far_dissolved_gas_concentration = self.params.far_dissolved_concentration_scaled
        liquid_fraction = self.calculate_liquid_fraction(solid_fraction)
        bubble_radius = self.calculate_bubble_radius(liquid_fraction=liquid_fraction)
        numerator = expansion_coefficient * (
            far_dissolved_gas_concentration
            - dissolved_gas_concentration * liquid_fraction
        )
        denominator = (
            self.params.stokes_rise_velocity_scaled
            * self.calculate_drag(bubble_radius=bubble_radius)
            + 1
        )
        return numerator / denominator

    def calculate_solid_fraction_derivative(
        self,
        temperature: Array,
        temperature_derivative: Array,
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -concentration_ratio
            * temperature_derivative
            / ((temperature - concentration_ratio) ** 2)
        )

    def calculate_temperature_second_derivative(
        self,
        temperature_derivative: Array,
        mushy_layer_depth: Array,
        solid_fraction_derivative: Array,
    ) -> Array:
        stefan_number = self.params.stefan_number
        return mushy_layer_depth * (
            temperature_derivative - stefan_number * solid_fraction_derivative
        )

    def calculate_frozen_gas_at_top(self) -> float:
        expansion_coefficient = self.params.expansion_coefficient
        far_dissolved_concentration_scaled = (
            self.params.far_dissolved_concentration_scaled
        )
        return far_dissolved_concentration_scaled * expansion_coefficient

    def check_volume_fractions_sum_to_one(self, solid_fraction, liquid_fraction):
        if (
            np.max(np.abs(solid_fraction + liquid_fraction - 1))
            > self.VOLUME_SUM_TOLERANCE
        ):
            return False
        return True

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
        solid_fraction = self.calculate_solid_fraction(temperature)
        gas_density = self.calculate_gas_density(temperature)
        gas_fraction = self.calculate_gas_fraction(
            solid_fraction,
            frozen_gas_fraction,
            gas_density,
            dissolved_gas_concentration,
        )
        liquid_fraction = self.calculate_liquid_fraction(solid_fraction)
        liquid_darcy_velocity = self.calculate_liquid_darcy_velocity(temperature)
        gas_darcy_velocity = self.calculate_gas_darcy_velocity(
            gas_fraction=gas_fraction, liquid_fraction=liquid_fraction
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


# TODO: use a setter to check small variation in results <03-01-23, Joe Fishlock> #
# def check_variation_is_small(array):
#     max_difference = np.max(np.abs(np.diff(array)))
#     if max_difference > DIFFERENCE_TOLERANCE:
#         return False
#     return True
