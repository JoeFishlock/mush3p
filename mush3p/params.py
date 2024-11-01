from __future__ import annotations
from typing import Dict, Any
import json
from dataclasses import dataclass, asdict

CELSIUS_TO_KELVIN = 273.15


@dataclass
class PhysicalParams:
    """Dimensional parameters for the three phase mushy layer system.

    This class implements the calculation of non-dimensional parameters for the system
    and provides the non_dimensionalise method to provide a NonDimensionalParams object.

    This class also provides save and load methods to serialize and deserialize to JSON.

    Args:
        name (str): Name of the simulation
        model_choice (str): Choice of model to use, either "full", "incompressible", "reduced" or "instant"
        bubble_radius (float): Radius of the bubbles [m]
        nucleation_time_scale (float): Time scale for bubble nucleation [s]
        far_salinity (float): Salinity of the far field [g/kg]
        far_temperature (float): Temperature of the far field [degC]
        far_dissolved_gas_concentration (float): Dissolved gas concentration in the far field [kg/kg]
        reference_velocity (float): Pulling speed [m/s], the default value is 1 micron/s
        liquid_density (float): Density of the liquid [kg/m3]
        gravitational_acceleration (float): Acceleration due to gravity [m/s2]
        liquid_dynamic_viscosity (float): Dynamic viscosity of the liquid [kg/m s],
            the default value gives the same kinematic viscosity used in Moreau et al 2014.
        surface_tension (float): Surface tension of the liquid [N/m]
        liquidus_slope (float): Slope of the liquidus curve [deg C / g/kg]
        eutectic_temperature (float): Eutectic temperature [deg C]
        latent_heat (float): Latent heat of fusion [J/kg]
        liquid_specific_heat_capacity (float): Specific heat capacity of the liquid [J/kg degC]
        solid_specific_heat_capacity (float): Specific heat capacity of the solid [J/kg degC]
        gas_specific_heat_capacity (float): Specific heat capacity of the gas [J/kg degC]
        liquid_thermal_conductivity (float): Thermal conductivity of the liquid [W/m degC]
        solid_thermal_conductivity (float): Thermal conductivity of the solid [W/m degC]
        gas_thermal_conductivity (float): Thermal conductivity of the gas [W/m degC]
        hele_shaw_gap_width (float): Width of the Hele-Shaw cell [m], the default value
            is the same value used in the experiments of Peppin et al 2007
        reference_permeability (float): Reference permeability [m2], the defulat value
            is the value used in Rees Jones and Worster 2014
        reference_pore_scale (float): Fitted pore throat radius at zero solid fraction [m],
            default value from Maus et al 2021
        pore_throat_exponent (float): Exponent for pore throat radius power law as a function of porosity,
            default value from Maus et al 2021
        reference_saturation_concentration (float): Saturation concentration for dissolved gas [kg/kg]
        specific_gas_constant (float): Specific gas constant [J/kg degK]
        atmospheric_pressure (float): Atmospheric pressure [Pa]
    """

    name: str
    model_choice: str = "full"
    bubble_radius: float = 1e-3
    nucleation_time_scale: float = 1000
    far_salinity: float = 35
    far_temperature: float = 0.1
    far_dissolved_gas_concentration: float = 3.71e-5
    reference_velocity: float = 1e-6
    liquid_density: float = 1028
    gravitational_acceleration: float = 9.81
    liquid_dynamic_viscosity: float = 2.78e-3
    surface_tension: float = 77e-3
    liquidus_slope: float = 0.07
    eutectic_temperature: float = -21.2
    latent_heat: float = 333e3
    liquid_specific_heat_capacity: float = 4209
    solid_specific_heat_capacity: float = 2108
    gas_specific_heat_capacity: float = 1004
    liquid_thermal_conductivity: float = 0.523
    solid_thermal_conductivity: float = 2.22
    gas_thermal_conductivity: float = 2e-2
    hele_shaw_gap_width: float = 5e-3
    reference_permeability: float = 1e-8
    reference_pore_scale: float = 1.95e-4
    pore_throat_exponent: float = 0.46
    reference_saturation_concentration: float = 3.71e-5
    specific_gas_constant: float = 286
    atmospheric_pressure: float = 1.01e5

    @property
    def initial_temperature(self) -> float:
        """Liquidus freezing temperature of the liquid at salinty far_salinity"""
        return -self.liquidus_slope * self.far_salinity

    @property
    def eutectic_salinity(self) -> float:
        """Calculated eutectic salinity from linear liquidus relation"""
        return -self.eutectic_temperature / self.liquidus_slope

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

    @classmethod
    def load(cls, filename: str) -> PhysicalParams:
        params = json.load(open(f"{filename}.json"))
        return cls(**params)

    def save(self, filename: str) -> None:
        json.dump(asdict(self), open(f"{filename}.json", "w"), indent=4)


@dataclass
class NonDimensionalParams:
    """Non-dimensional parameters for the three phase mushy layer system.

    Note: these can be initialised directly or by using the non_dimensionalise method of PhysicalParams.

    This class also provides save and load methods to serialize and deserialize to JSON.

    Args:
        name (str): Name of the simulation
        model_choice (str): Choice of model to use, either "full", "incompressible", "reduced" or "instant"
        concentration_ratio (float): Ratio of the far field salinity to the salinity difference
        stefan_number (float): Ratio of the latent heat to the sensible heat
        hele_shaw_permeability_scaled (float): Ratio of the permeability of the Hele-Shaw cell to the reference permeability
        far_temperature_scaled (float): Non-dimensionalised far field temperature
        damkholer_number (float): Ratio of the thermal time scale to the nucleation time scale
        expansion_coefficient (float): Relative volume increase on exsolving a saturation amount of dissolved gas
        stokes_rise_velocity_scaled (float): Ratio of the bubble Stokes' rise velocity to the reference velocity
        bubble_radius_scaled (float): Ratio of the bubble radius to the pore throat radius
        pore_throat_exponent (float): Power law exponent for pore throat radius as a function of porosity
        far_dissolved_concentration_scaled (float): Ratio of the far field dissolved gas concentration to the saturation concentration
    """

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

    @classmethod
    def load(cls, filename: str) -> NonDimensionalParams:
        params = json.load(open(f"{filename}.json"))
        return cls(**params)

    def save(self, filename: str) -> None:
        json.dump(asdict(self), open(f"{filename}.json", "w"), indent=4)
