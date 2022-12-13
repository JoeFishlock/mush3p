from __future__ import annotations
from typing import Protocol
import os
import json
from dataclasses import dataclass
from agate.model import Model
from agate.output import NonDimensionalOutput


class NumericalParameters:
    tol: float = 1e-7


class NonDimensionalParams(Protocol):
    def solve(
        self, numerical_parameters: NumericalParameters, model: Model
    ) -> NonDimensionalOutput:
        ...


class PhysicalParams(Protocol):
    @property
    def params(self) -> dict:
        ...

    @classmethod
    def load(cls, filename: str) -> PhysicalParams:
        ...

    def save(self, filename: str) -> None:
        ...

    def non_dimensionalise(self) -> NonDimensionalParams:
        ...


@dataclass
class FullPhysicalParams:
    name: str
    liquid_density: float = 1028
    solid_density: float = 998

    @property
    def params(self) -> dict:
        return {
            "name": self.name,
            "liquid_density": self.liquid_density,
            "solid_density": self.solid_density,
        }

    @classmethod
    def load(cls, filename: str) -> FullPhysicalParams:
        data_path = "data"
        params = json.load(open(f"{data_path}/{filename}.json"))
        return cls(**params)

    def save(self, filename: str) -> None:
        data_path = "data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        json.dump(self.params, open(f"{data_path}/{filename}.json", "w"))

    def non_dimensionalise(self) -> FullNonDimensionalParams:
        non_dimensional_params = {
            "name": self.name,
            "concentration_ratio": self.concentration_ratio,
            "stefan_number": self.stefan_number,
            "hele_shaw_permeability": self.hele_shaw_permeability,
            "far_temperature": self.far_temperature,
            "damkholer_number": self.damkholer_number,
            "expansion_coefficient": self.expansion_coefficient,
            "stokes_rise_velocity": self.stokes_rise_velocity,
            "bubble_radius": self.bubble_radius,
            "far_concentration": self.far_concentration,
            "gas_conductivity": self.gas_conductivity,
            "hydrostatic_pressure": self.hydrostatic_pressure,
            "laplace_pressure": self.laplace_pressure,
            "kelvin_conversion_temperature": self.kelvin_conversion_temperature,
            "atmospheric_pressure": self.atmospheric_pressure,
        }
        return FullNonDimensionalParams(**non_dimensional_params)


@dataclass
class FullNonDimensionalParams:
    name: str

    # mushy layer params
    concentration_ratio: float
    stefan_number: float
    hele_shaw_permeability: float
    far_temperature: float

    # gas params
    damkholer_number: float
    expansion_coefficient: float
    stokes_rise_velocity: float
    bubble_radius: float
    far_concentration: float
    gas_conductivity: float

    # compressible gas params
    hydrostatic_pressure: float
    laplace_pressure: float
    kelvin_conversion_temperature: float
    atmospheric_pressure: float
