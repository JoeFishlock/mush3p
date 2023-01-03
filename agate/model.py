"""
model.py contains the equations for solving the full model

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
from scipy.optimize import fsolve  # type: ignore
from numpy.typing import NDArray

Array = Union[NDArray, float]


class FullModel:
    """Class containing full equations for system"""

    # From Maus paper
    PORE_THROAT_SCALING: float = 0.5
    DRAG_EXPONENT: int = 6
    GAS_FRACTION_GUESS: float = 0.01
    # Initial Conditions for solver
    INITIAL_MESH_NODES: int = 20
    INITIAL_HEIGHT: NDArray = np.linspace(-1, 0, INITIAL_MESH_NODES)
    INITIAL_TEMPERATURE: NDArray = np.linspace(0, -1, INITIAL_MESH_NODES)
    INITIAL_TEMPERATURE_DERIVATIVE = np.full_like(INITIAL_TEMPERATURE, -1.0)
    INITIAL_DISSOLVED_GAS_CONCENTRATION = np.linspace(0.8, 1.0, INITIAL_MESH_NODES)
    INITIAL_HYDROSTATIC_PRESSURE = np.linspace(-0.1, 0, INITIAL_MESH_NODES)
    INITIAL_FROZEN_GAS_FRACTION = np.full_like(INITIAL_TEMPERATURE, 0.02)
    INITIAL_MUSHY_LAYER_DEPTH = np.full_like(INITIAL_TEMPERATURE, 1.5)

    INITIAL_VARIABLES = np.vstack(
        (
            INITIAL_TEMPERATURE,
            INITIAL_TEMPERATURE_DERIVATIVE,
            INITIAL_DISSOLVED_GAS_CONCENTRATION,
            INITIAL_HYDROSTATIC_PRESSURE,
            INITIAL_FROZEN_GAS_FRACTION,
            INITIAL_MUSHY_LAYER_DEPTH,
        )
    )

    # Tolerances for error checking
    DIFFERENCE_TOLERANCE = 1e-8
    VOLUME_SUM_TOLERANCE = 1e-8

    def __init__(self, params) -> None:
        self.params = params

    def calculate_solid_salinity(self, temperature: Array) -> Array:
        return np.full_like(temperature, -self.params.concentration_ratio)

    def calculate_liquid_salinity(self, temperature: Array) -> Array:
        return -temperature

    def calculate_liquid_darcy_velocity(
        self, gas_fraction: Array, frozen_gas_fraction: Array
    ) -> Array:
        return gas_fraction - frozen_gas_fraction

    def calculate_solid_fraction(
        self, temperature: Array, frozen_gas_fraction: Array
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -(1 - frozen_gas_fraction)
            * temperature
            / (concentration_ratio - temperature)
        )

    def calculate_liquid_fraction(
        self, solid_fraction: Array, gas_fraction: Array
    ) -> Array:
        return 1 - solid_fraction - gas_fraction

    def calculate_liquid_saturation(
        self, solid_fraction: Array, liquid_fraction: Array
    ) -> Array:
        return liquid_fraction / (1 - solid_fraction)

    def calculate_bubble_radius(self, liquid_fraction: Array) -> Array:
        exponent = self.PORE_THROAT_SCALING
        return self.params.bubble_radius_scaled / (liquid_fraction**exponent)

    def calculate_lag(self, bubble_radius: Array) -> Array:
        lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
        lag = np.where(bubble_radius > 1, 0.5, lag)
        return lag

    def calculate_drag(self, bubble_radius: Array) -> Array:
        exponent = self.DRAG_EXPONENT
        drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** exponent)
        drag = np.where(bubble_radius > 1, 0, drag)
        return drag

    def calculate_gas_darcy_velocity(
        self,
        gas_fraction: Array,
        liquid_fraction: Array,
        liquid_darcy_velocity: Array,
    ) -> Array:
        bubble_radius = self.calculate_bubble_radius(liquid_fraction=liquid_fraction)
        drag = self.calculate_drag(bubble_radius=bubble_radius)
        lag = self.calculate_lag(bubble_radius=bubble_radius)

        buoyancy_term = self.params.stokes_rise_velocity_scaled * drag
        liquid_term = 2 * lag * liquid_darcy_velocity / liquid_fraction

        return gas_fraction * (buoyancy_term + liquid_term)

    def calculate_gas_density(
        self,
        temperature: Array,
        hydrostatic_pressure: Array,
        mushy_layer_depth: Array,
        height: Array,
    ) -> Array:
        kelvin = self.params.kelvin_conversion_temperature
        temperature_term = (1 + temperature / kelvin) ** (-1)
        pressure_term = hydrostatic_pressure / self.params.atmospheric_pressure_scaled
        laplace_term = self.params.laplace_pressure_scale
        depth_term = (
            -self.params.hydrostatic_pressure_scale * height * mushy_layer_depth
        )

        return temperature_term * (1 + pressure_term + laplace_term + depth_term)

    def calculate_gas_fraction(
        self,
        solid_fraction,
        frozen_gas_fraction: Array,
        gas_density: Array,
        dissolved_gas_concentration: Array,
    ) -> Any:
        expansion_coefficient = self.params.expansion_coefficient
        far_dissolved_gas_concentration = self.params.far_dissolved_concentration_scaled

        def residual(gas_fraction: Array) -> Array:
            liquid_darcy_velocity = self.calculate_liquid_darcy_velocity(
                gas_fraction=gas_fraction, frozen_gas_fraction=frozen_gas_fraction
            )

            liquid_fraction = self.calculate_liquid_fraction(
                solid_fraction=solid_fraction, gas_fraction=gas_fraction
            )

            gas_darcy_velocity = self.calculate_gas_darcy_velocity(
                gas_fraction=gas_fraction,
                liquid_fraction=liquid_fraction,
                liquid_darcy_velocity=liquid_darcy_velocity,
            )

            return (
                gas_density
                * (gas_fraction + gas_darcy_velocity)
                / expansion_coefficient
                + dissolved_gas_concentration
                * (liquid_fraction + liquid_darcy_velocity)
                - far_dissolved_gas_concentration * (1 - frozen_gas_fraction)
            )

        # TODO: write a unit test to test gas_fraction is 0 when no dissolved gas present
        # <14-12-22, Joe Fishlock> #

        initial_guess = np.full_like(solid_fraction, self.GAS_FRACTION_GUESS)
        return fsolve(residual, initial_guess)

    def calculate_permeability(self, liquid_fraction: Array) -> Array:
        liquid_permeability_reciprocal = (
            1 - liquid_fraction
        ) ** 2 / liquid_fraction**3
        reference = self.params.hele_shaw_permeability_scaled
        return ((1 / reference) + liquid_permeability_reciprocal) ** (-1)

    def calculate_saturation_concentration(self, temperature: Array) -> Array:
        return np.full_like(temperature, 1)

    def calculate_unconstrained_nucleation_rate(
        self, dissolved_gas_concentration: Array, saturation_concentration: Array
    ) -> Array:
        return dissolved_gas_concentration - saturation_concentration

    def calculate_nucleation_indicator(
        self, dissolved_gas_concentration: Array, saturation_concentration: Array
    ) -> Array:
        return np.where(dissolved_gas_concentration >= saturation_concentration, 1, 0)

    def calculate_nucleation_rate(
        self, temperature: Array, dissolved_gas_concentration: Array
    ) -> Array:
        saturation_concentration = self.calculate_saturation_concentration(
            temperature=temperature
        )
        unconstrained_nucleation_rate = self.calculate_unconstrained_nucleation_rate(
            dissolved_gas_concentration=dissolved_gas_concentration,
            saturation_concentration=saturation_concentration,
        )
        nucleation_indicator = self.calculate_nucleation_indicator(
            dissolved_gas_concentration=dissolved_gas_concentration,
            saturation_concentration=saturation_concentration,
        )

        return nucleation_indicator * unconstrained_nucleation_rate

    def calculate_solid_fraction_derivative(
        self,
        temperature: Array,
        temperature_derivative: Array,
        frozen_gas_fraction: Array,
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -concentration_ratio
            * (1 - frozen_gas_fraction)
            * temperature_derivative
            / (concentration_ratio - temperature) ** 2
        )

    def calculate_gas_fraction_derivative(
        self, gas_fraction: Array, height: Array
    ) -> Array:
        """Numerically approximate the derivative with finite difference."""
        return np.gradient(gas_fraction, height)

    def calculate_hydrostatic_pressure_derivative(
        self,
        permeability: Array,
        liquid_darcy_velocity: Array,
        mushy_layer_depth: Array,
    ) -> Array:
        return -mushy_layer_depth * liquid_darcy_velocity / permeability

    def calculate_temperature_derivative(
        self,
        temperature_derivative: Array,
    ) -> Array:
        return temperature_derivative

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
        gas_conductivity_ratio = self.params.gas_conductivity_ratio

        heating = (
            mushy_layer_depth * (1 - frozen_gas_fraction) * temperature_derivative
            - mushy_layer_depth * stefan_number * solid_fraction_derivative
        )

        gas_insulation = (
            (1 - gas_conductivity_ratio)
            * gas_fraction_derivative
            * temperature_derivative
        )

        return (heating + gas_insulation) / (
            1 - (1 - gas_conductivity_ratio) * gas_fraction
        )

    def calculate_dissolved_gas_concentration_derivative(
        self,
        dissolved_gas_concentration: Array,
        solid_fraction_derivative: Array,
        frozen_gas_fraction: Array,
        solid_fraction: Array,
        mushy_layer_depth,
        nucleation_rate: Array,
    ) -> Array:

        damkholer_number = self.params.damkholer_number
        freezing = dissolved_gas_concentration * solid_fraction_derivative
        dissolution = -damkholer_number * mushy_layer_depth * nucleation_rate

        return (freezing + dissolution) / (1 - frozen_gas_fraction - solid_fraction)

    def calculate_zero_derivative(
        self,
        temperature: Array,
    ) -> Array:
        return np.zeros_like(temperature)

    def calculate_frozen_gas_at_top(self, gas_density_at_top: float) -> float:
        expansion_coefficient = self.params.expansion_coefficient
        far_dissolved_concentration_scaled = (
            self.params.far_dissolved_concentration_scaled
        )
        return (
            1
            + gas_density_at_top
            / (expansion_coefficient * far_dissolved_concentration_scaled)
        ) ** (-1)

    def check_volume_fractions_sum_to_one(
        self, solid_fraction, liquid_fraction, gas_fraction
    ):
        if (
            np.max(np.abs(solid_fraction + liquid_fraction + gas_fraction - 1))
            > self.VOLUME_SUM_TOLERANCE
        ):
            return False
        return True

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
            hydrostatic_pressure=hydrostatic_pressure,
            mushy_layer_depth=mushy_layer_depth,
            height=height,
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
            hydrostatic_pressure=hydrostatic_pressure_at_top,
            mushy_layer_depth=mushy_layer_depth_at_top,
            height=0,
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
        gas_density = self.calculate_gas_density(
            temperature, hydrostatic_pressure, mushy_layer_depth, height
        )
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


class IncompressibleModel:
    """Class containing equations with no gas compressibility.
    The non dimensional gas density is set to 1.0"""

    # From Maus paper
    PORE_THROAT_SCALING: float = 0.5
    DRAG_EXPONENT: int = 6
    GAS_FRACTION_GUESS: float = 0.01
    # Initial Conditions for solver
    INITIAL_MESH_NODES: int = 20
    INITIAL_HEIGHT: NDArray = np.linspace(-1, 0, INITIAL_MESH_NODES)
    INITIAL_TEMPERATURE: NDArray = np.linspace(0, -1, INITIAL_MESH_NODES)
    INITIAL_TEMPERATURE_DERIVATIVE = np.full_like(INITIAL_TEMPERATURE, -1.0)
    INITIAL_DISSOLVED_GAS_CONCENTRATION = np.linspace(0.8, 1.0, INITIAL_MESH_NODES)
    INITIAL_HYDROSTATIC_PRESSURE = np.linspace(-0.1, 0, INITIAL_MESH_NODES)
    INITIAL_FROZEN_GAS_FRACTION = np.full_like(INITIAL_TEMPERATURE, 0.02)
    INITIAL_MUSHY_LAYER_DEPTH = np.full_like(INITIAL_TEMPERATURE, 1.5)

    INITIAL_VARIABLES = np.vstack(
        (
            INITIAL_TEMPERATURE,
            INITIAL_TEMPERATURE_DERIVATIVE,
            INITIAL_DISSOLVED_GAS_CONCENTRATION,
            INITIAL_HYDROSTATIC_PRESSURE,
            INITIAL_FROZEN_GAS_FRACTION,
            INITIAL_MUSHY_LAYER_DEPTH,
        )
    )

    # Tolerances for error checking
    DIFFERENCE_TOLERANCE = 1e-8
    VOLUME_SUM_TOLERANCE = 1e-8

    def __init__(self, params) -> None:
        self.params = params

    def calculate_solid_salinity(self, temperature: Array) -> Array:
        return np.full_like(temperature, -self.params.concentration_ratio)

    def calculate_liquid_salinity(self, temperature: Array) -> Array:
        return -temperature

    def calculate_liquid_darcy_velocity(
        self, gas_fraction: Array, frozen_gas_fraction: Array
    ) -> Array:
        return gas_fraction - frozen_gas_fraction

    def calculate_solid_fraction(
        self, temperature: Array, frozen_gas_fraction: Array
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -(1 - frozen_gas_fraction)
            * temperature
            / (concentration_ratio - temperature)
        )

    def calculate_liquid_fraction(
        self, solid_fraction: Array, gas_fraction: Array
    ) -> Array:
        return 1 - solid_fraction - gas_fraction

    def calculate_liquid_saturation(
        self, solid_fraction: Array, liquid_fraction: Array
    ) -> Array:
        return liquid_fraction / (1 - solid_fraction)

    def calculate_bubble_radius(self, liquid_fraction: Array) -> Array:
        exponent = self.PORE_THROAT_SCALING
        return self.params.bubble_radius_scaled / (liquid_fraction**exponent)

    def calculate_lag(self, bubble_radius: Array) -> Array:
        lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
        lag = np.where(bubble_radius > 1, 0.5, lag)
        return lag

    def calculate_drag(self, bubble_radius: Array) -> Array:
        exponent = self.DRAG_EXPONENT
        drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** exponent)
        drag = np.where(bubble_radius > 1, 0, drag)
        return drag

    def calculate_gas_darcy_velocity(
        self,
        gas_fraction: Array,
        liquid_fraction: Array,
        liquid_darcy_velocity: Array,
    ) -> Array:
        bubble_radius = self.calculate_bubble_radius(liquid_fraction=liquid_fraction)
        drag = self.calculate_drag(bubble_radius=bubble_radius)
        lag = self.calculate_lag(bubble_radius=bubble_radius)

        buoyancy_term = self.params.stokes_rise_velocity_scaled * drag
        liquid_term = 2 * lag * liquid_darcy_velocity / liquid_fraction

        return gas_fraction * (buoyancy_term + liquid_term)

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

        def residual(gas_fraction: Array) -> Array:
            liquid_darcy_velocity = self.calculate_liquid_darcy_velocity(
                gas_fraction=gas_fraction, frozen_gas_fraction=frozen_gas_fraction
            )

            liquid_fraction = self.calculate_liquid_fraction(
                solid_fraction=solid_fraction, gas_fraction=gas_fraction
            )

            gas_darcy_velocity = self.calculate_gas_darcy_velocity(
                gas_fraction=gas_fraction,
                liquid_fraction=liquid_fraction,
                liquid_darcy_velocity=liquid_darcy_velocity,
            )

            return (
                gas_density
                * (gas_fraction + gas_darcy_velocity)
                / expansion_coefficient
                + dissolved_gas_concentration
                * (liquid_fraction + liquid_darcy_velocity)
                - far_dissolved_gas_concentration * (1 - frozen_gas_fraction)
            )

        initial_guess = np.full_like(solid_fraction, self.GAS_FRACTION_GUESS)
        return fsolve(residual, initial_guess)

    def calculate_permeability(self, liquid_fraction: Array) -> Array:
        liquid_permeability_reciprocal = (
            1 - liquid_fraction
        ) ** 2 / liquid_fraction**3
        reference = self.params.hele_shaw_permeability_scaled
        return ((1 / reference) + liquid_permeability_reciprocal) ** (-1)

    def calculate_saturation_concentration(self, temperature: Array) -> Array:
        return np.full_like(temperature, 1)

    def calculate_unconstrained_nucleation_rate(
        self, dissolved_gas_concentration: Array, saturation_concentration: Array
    ) -> Array:
        return dissolved_gas_concentration - saturation_concentration

    def calculate_nucleation_indicator(
        self, dissolved_gas_concentration: Array, saturation_concentration: Array
    ) -> Array:
        return np.where(dissolved_gas_concentration >= saturation_concentration, 1, 0)

    def calculate_nucleation_rate(
        self, temperature: Array, dissolved_gas_concentration: Array
    ) -> Array:
        saturation_concentration = self.calculate_saturation_concentration(
            temperature=temperature
        )
        unconstrained_nucleation_rate = self.calculate_unconstrained_nucleation_rate(
            dissolved_gas_concentration=dissolved_gas_concentration,
            saturation_concentration=saturation_concentration,
        )
        nucleation_indicator = self.calculate_nucleation_indicator(
            dissolved_gas_concentration=dissolved_gas_concentration,
            saturation_concentration=saturation_concentration,
        )

        return nucleation_indicator * unconstrained_nucleation_rate

    def calculate_solid_fraction_derivative(
        self,
        temperature: Array,
        temperature_derivative: Array,
        frozen_gas_fraction: Array,
    ) -> Array:
        concentration_ratio = self.params.concentration_ratio
        return (
            -concentration_ratio
            * (1 - frozen_gas_fraction)
            * temperature_derivative
            / (concentration_ratio - temperature) ** 2
        )

    def calculate_gas_fraction_derivative(
        self, gas_fraction: Array, height: Array
    ) -> Array:
        """Numerically approximate the derivative with finite difference."""
        return np.gradient(gas_fraction, height)

    def calculate_hydrostatic_pressure_derivative(
        self,
        permeability: Array,
        liquid_darcy_velocity: Array,
        mushy_layer_depth: Array,
    ) -> Array:
        return -mushy_layer_depth * liquid_darcy_velocity / permeability

    def calculate_temperature_derivative(
        self,
        temperature_derivative: Array,
    ) -> Array:
        return temperature_derivative

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
        gas_conductivity_ratio = self.params.gas_conductivity_ratio

        heating = (
            mushy_layer_depth * (1 - frozen_gas_fraction) * temperature_derivative
            - mushy_layer_depth * stefan_number * solid_fraction_derivative
        )

        gas_insulation = (
            (1 - gas_conductivity_ratio)
            * gas_fraction_derivative
            * temperature_derivative
        )

        return (heating + gas_insulation) / (
            1 - (1 - gas_conductivity_ratio) * gas_fraction
        )

    def calculate_dissolved_gas_concentration_derivative(
        self,
        dissolved_gas_concentration: Array,
        solid_fraction_derivative: Array,
        frozen_gas_fraction: Array,
        solid_fraction: Array,
        mushy_layer_depth,
        nucleation_rate: Array,
    ) -> Array:

        damkholer_number = self.params.damkholer_number
        freezing = dissolved_gas_concentration * solid_fraction_derivative
        dissolution = -damkholer_number * mushy_layer_depth * nucleation_rate

        return (freezing + dissolution) / (1 - frozen_gas_fraction - solid_fraction)

    def calculate_zero_derivative(
        self,
        temperature: Array,
    ) -> Array:
        return np.zeros_like(temperature)

    def calculate_frozen_gas_at_top(self, gas_density_at_top: float) -> float:
        expansion_coefficient = self.params.expansion_coefficient
        far_dissolved_concentration_scaled = (
            self.params.far_dissolved_concentration_scaled
        )
        return (
            1
            + gas_density_at_top
            / (expansion_coefficient * far_dissolved_concentration_scaled)
        ) ** (-1)

    def check_volume_fractions_sum_to_one(
        self, solid_fraction, liquid_fraction, gas_fraction
    ):
        if (
            np.max(np.abs(solid_fraction + liquid_fraction + gas_fraction - 1))
            > self.VOLUME_SUM_TOLERANCE
        ):
            return False
        return True

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


# TODO: use a setter to check small variation in results <03-01-23, Joe Fishlock> #
# def check_variation_is_small(array):
#     max_difference = np.max(np.abs(np.diff(array)))
#     if max_difference > DIFFERENCE_TOLERANCE:
#         return False
#     return True


# Put all models that can be run in this dictionary
MODEL_OPTIONS = {"full": FullModel, "incompressible": IncompressibleModel}
