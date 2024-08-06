from __future__ import annotations
from typing import Dict, Any
import json
from dataclasses import dataclass, asdict

CELSIUS_TO_KELVIN = 273.15


@dataclass
class PhysicalParams:
    name: str
    model_choice: str = "full"
    liquid_density: float = 1028  # kg/m3
    far_salinity: float = 34  # psu (g/kg)
    eutectic_salinity: float = 230  # psu (g/kg)
    far_temperature: float = 0.1  # degC
    initial_temperature: float = -2  # degC
    eutectic_temperature: float = -21  # deg C
    latent_heat: float = 333.4e3  # J/kg
    liquid_specific_heat_capacity: float = 4209  # J/kg degC
    solid_specific_heat_capacity: float = 2108  # J/kg degC
    gas_specific_heat_capacity: float = 1004  # J/kg degC
    hele_shaw_gap_width: float = 1e-3  # m
    reference_permeability: float = 1e-8  # m2
    nucleation_time_scale: float = 250  # s
    reference_saturation_concentration: float = 3.71e-5  # kg/kg
    far_dissolved_gas_concentration: float = 3.71e-5  # kg/kg
    specific_gas_constant: float = 286  # J/kg degK
    reference_velocity: float = 3e-6  # m/s
    bubble_radius: float = 1e-3  # m
    reference_pore_scale: float = 1e-3  # m
    pore_throat_exponent: float = 0.46
    gravitational_acceleration: float = 9.81  # m/s2
    liquid_dynamic_viscosity: float = 1.906e-3  # kg/m s
    liquid_thermal_conductivity: float = 0.523  # W/m degC
    solid_thermal_conductivity: float = 2.22  # W/m degC
    gas_thermal_conductivity: float = 2e-2  # W/m degC
    surface_tension: float = 77.09e-3  # N/m
    atmospheric_pressure: float = 1.01e5  # Pa

    @property
    def liquid_thermal_diffusivity(self) -> float:
        return self.liquid_thermal_conductivity / (
            self.liquid_density * self.liquid_specific_heat_capacity
        )

    @property
    def length_scale(self) -> float:
        return self.liquid_thermal_diffusivity / self.reference_velocity

    @property
    def time_scale(self) -> float:
        return self.liquid_thermal_diffusivity / self.reference_velocity**2

    @property
    def reference_gas_density(self) -> float:
        return self.atmospheric_pressure / (
            self.specific_gas_constant * (self.initial_temperature + CELSIUS_TO_KELVIN)
        )

    @property
    def gas_density_ratio(self) -> float:
        return self.reference_gas_density / self.liquid_density

    @property
    def pressure_scale(self) -> float:
        return (
            self.liquid_thermal_diffusivity
            * self.liquid_dynamic_viscosity
            / self.reference_permeability
        )

    @property
    def params(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, filename: str) -> PhysicalParams:
        params = json.load(open(f"{filename}.json"))
        return cls(**params)

    def save(self, filename: str) -> None:
        json.dump(self.params, open(f"{filename}.json", "w"), indent=4)

    @property
    def concentration_ratio(self) -> float:
        salinity_diff = self.eutectic_salinity - self.far_salinity
        return self.far_salinity / salinity_diff

    @property
    def stefan_number(self) -> float:
        temperature_diff = self.initial_temperature - self.eutectic_temperature
        return self.latent_heat / (
            temperature_diff * self.liquid_specific_heat_capacity
        )

    @property
    def hele_shaw_permeability_scaled(self) -> float:
        return self.hele_shaw_gap_width**2 / (12 * self.reference_permeability)

    @property
    def far_temperature_scaled(self) -> float:
        return (self.far_temperature - self.initial_temperature) / (
            self.initial_temperature - self.eutectic_temperature
        )

    @property
    def damkholer_number(self) -> float:
        return self.time_scale / self.nucleation_time_scale

    @property
    def expansion_coefficient(self) -> float:
        return (
            self.liquid_density * self.reference_saturation_concentration
        ) / self.reference_gas_density

    @property
    def stokes_rise_velocity_scaled(self) -> float:
        return (
            self.liquid_density
            * self.gravitational_acceleration
            * self.reference_pore_scale**2
        ) / (3 * self.liquid_dynamic_viscosity * self.reference_velocity)

    @property
    def bubble_radius_scaled(self) -> float:
        return self.bubble_radius / self.reference_pore_scale

    @property
    def far_dissolved_concentration_scaled(self) -> float:
        return (
            self.far_dissolved_gas_concentration
            / self.reference_saturation_concentration
        )

    @property
    def gas_conductivity_ratio(self) -> float:
        return self.gas_thermal_conductivity / self.liquid_thermal_conductivity

    @property
    def solid_conductivity_ratio(self) -> float:
        return self.solid_thermal_conductivity / self.liquid_thermal_conductivity

    @property
    def solid_specific_heat_capacity_ratio(self) -> float:
        return self.solid_specific_heat_capacity / self.liquid_specific_heat_capacity

    @property
    def gas_specific_heat_capacity_ratio(self) -> float:
        return self.gas_specific_heat_capacity / self.liquid_specific_heat_capacity

    @property
    def hydrostatic_pressure_scale(self) -> float:
        return (
            self.liquid_density
            * self.gravitational_acceleration
            * self.liquid_thermal_diffusivity
        ) / (self.atmospheric_pressure * self.reference_velocity)

    @property
    def laplace_pressure_scale(self) -> float:
        return (
            2 * self.surface_tension / (self.bubble_radius * self.atmospheric_pressure)
        )

    @property
    def kelvin_conversion_temperature(self) -> float:
        return (self.initial_temperature + CELSIUS_TO_KELVIN) / (
            self.initial_temperature - self.eutectic_temperature
        )

    @property
    def atmospheric_pressure_scaled(self) -> float:
        return self.atmospheric_pressure / self.pressure_scale

    def non_dimensionalise(self) -> NonDimensionalParams:
        non_dimensional_params: Dict[str, Any] = {
            "name": self.name,
            "model_choice": self.model_choice,
            "concentration_ratio": self.concentration_ratio,
            "stefan_number": self.stefan_number,
            "hele_shaw_permeability_scaled": self.hele_shaw_permeability_scaled,
            "far_temperature_scaled": self.far_temperature_scaled,
            "damkholer_number": self.damkholer_number,
            "expansion_coefficient": self.expansion_coefficient,
            "stokes_rise_velocity_scaled": self.stokes_rise_velocity_scaled,
            "bubble_radius_scaled": self.bubble_radius_scaled,
            "pore_throat_exponent": self.pore_throat_exponent,
            "far_dissolved_concentration_scaled": self.far_dissolved_concentration_scaled,
            "gas_conductivity_ratio": self.gas_conductivity_ratio,
            "solid_conductivity_ratio": self.solid_conductivity_ratio,
            "solid_specific_heat_capacity_ratio": self.solid_specific_heat_capacity_ratio,
            "gas_specific_heat_capacity_ratio": self.gas_specific_heat_capacity_ratio,
            "hydrostatic_pressure_scale": self.hydrostatic_pressure_scale,
            "laplace_pressure_scale": self.laplace_pressure_scale,
            "kelvin_conversion_temperature": self.kelvin_conversion_temperature,
            "atmospheric_pressure_scaled": self.atmospheric_pressure_scaled,
            "gas_density_ratio": self.gas_density_ratio,
        }
        return NonDimensionalParams(**non_dimensional_params)


@dataclass
class NonDimensionalParams:
    name: str
    model_choice: str

    # mushy layer params
    concentration_ratio: float
    stefan_number: float
    hele_shaw_permeability_scaled: float
    far_temperature_scaled: float
    solid_conductivity_ratio: float
    solid_specific_heat_capacity_ratio: float
    gas_specific_heat_capacity_ratio: float

    # gas params
    damkholer_number: float
    expansion_coefficient: float
    stokes_rise_velocity_scaled: float
    bubble_radius_scaled: float
    pore_throat_exponent: float
    far_dissolved_concentration_scaled: float
    gas_conductivity_ratio: float
    gas_density_ratio: float

    # compressible gas params
    hydrostatic_pressure_scale: float
    laplace_pressure_scale: float
    kelvin_conversion_temperature: float
    atmospheric_pressure_scaled: float

    @property
    def params(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, filename: str) -> NonDimensionalParams:
        params = json.load(open(f"{filename}.json"))
        return cls(**params)

    def save(self, filename: str) -> None:
        json.dump(self.params, open(f"{filename}.json", "w"), indent=4)
