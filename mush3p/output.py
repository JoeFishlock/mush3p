"""Class for storing simulation output"""
import numpy as np
import json
from dataclasses import asdict
from .params import NonDimensionalParams
from .model import MODEL_OPTIONS


class NonDimensionalResults:
    """class to store non-dimensional results of a simulation"""

    def __init__(
        self,
        non_dimensional_parameters,
        temperature_array,
        temperature_derivative_array,
        concentration_array,
        hydrostatic_pressure_array,
        frozen_gas_fraction,
        mushy_layer_depth,
        height_array,
    ):
        self.name = non_dimensional_parameters.name
        self.params = non_dimensional_parameters

        self.temperature_array = np.array(temperature_array)
        self.temperature_derivative_array = np.array(temperature_derivative_array)
        self.concentration_array = np.array(concentration_array)
        self.hydrostatic_pressure_array = np.array(hydrostatic_pressure_array)
        self.frozen_gas_fraction = frozen_gas_fraction
        self.mushy_layer_depth = mushy_layer_depth
        self.height_array = np.array(height_array)

        # Calculate all mushy layer arrays
        if self.params.model_choice == "instant":
            model = MODEL_OPTIONS[self.params.model_choice](
                self.params,
                self.height_array,
                self.temperature_array,
                self.temperature_derivative_array,
                self.hydrostatic_pressure_array,
                self.frozen_gas_fraction,
                self.mushy_layer_depth,
            )
        else:
            model = MODEL_OPTIONS[self.params.model_choice](
                self.params,
                self.height_array,
                self.temperature_array,
                self.temperature_derivative_array,
                self.concentration_array,
                self.hydrostatic_pressure_array,
                self.frozen_gas_fraction,
                self.mushy_layer_depth,
            )
        self.solid_salinity_array = model.solid_salinity
        self.liquid_salinity_array = model.liquid_salinity
        self.solid_fraction_array = model.solid_fraction
        self.liquid_fraction_array = model.liquid_fraction
        self.gas_fraction_array = model.gas_fraction
        self.gas_density_array = model.gas_density
        self.liquid_darcy_velocity_array = model.liquid_darcy_velocity
        self.gas_darcy_velocity_array = model.gas_darcy_velocity

    def save(self, filename: str) -> None:
        data = {
            "non_dimensional_parameters": asdict(self.params),
            "temperature_array": self.temperature_array.tolist(),
            "temperature_derivative_array": self.temperature_derivative_array.tolist(),
            "concentration_array": self.concentration_array.tolist(),
            "hydrostatic_pressure_array": self.hydrostatic_pressure_array.tolist(),
            "frozen_gas_fraction": self.frozen_gas_fraction,
            "mushy_layer_depth": self.mushy_layer_depth,
            "height_array": self.height_array.tolist(),
        }
        with open(f"{filename}.json", "w") as fp:
            json.dump(data, fp, indent=4)

    @classmethod
    def load(cls, filename: str):
        with open(f"{filename}.json", "r") as fp:
            data = json.load(fp)
        params = data["non_dimensional_parameters"]
        data["non_dimensional_parameters"] = NonDimensionalParams(**params)
        return cls(**data)

    def liquid_salinity(self, height):
        return np.interp(
            height, self.height_array, self.liquid_salinity_array, right=np.NaN
        )

    def temperature(self, height):
        liquid_darcy_velocity_at_bottom = self.liquid_darcy_velocity_array[0]
        liquid_range = np.linspace(-10, -1.1, 100)
        liquid_values = self.params.far_temperature_scaled * (
            1
            - np.exp(
                (1 + liquid_darcy_velocity_at_bottom)
                * self.mushy_layer_depth
                * (liquid_range + 1)
            )
        )
        return np.interp(
            height,
            np.hstack((liquid_range, self.height_array)),
            np.hstack((liquid_values, self.temperature_array)),
            left=self.params.far_temperature_scaled,
        )

    def concentration(self, height):
        return np.interp(
            height, self.height_array, self.concentration_array, right=np.NaN
        )

    def hydrostatic_pressure(self, height):
        return np.interp(
            height, self.height_array, self.hydrostatic_pressure_array, right=np.NaN
        )

    def solid_fraction(self, height):
        return np.interp(
            height,
            self.height_array,
            self.solid_fraction_array,
            right=1 - self.frozen_gas_fraction,
        )

    def liquid_fraction(self, height):
        return np.interp(height, self.height_array, self.liquid_fraction_array, right=0)

    def gas_fraction(self, height):
        return np.interp(
            height,
            self.height_array,
            self.gas_fraction_array,
            left=0,
            right=self.frozen_gas_fraction,
        )

    def liquid_darcy_velocity(self, height):
        return np.interp(
            height, self.height_array, self.liquid_darcy_velocity_array, right=np.NaN
        )

    def gas_darcy_velocity(self, height):
        return np.interp(
            height,
            self.height_array,
            self.gas_darcy_velocity_array,
            left=np.NaN,
            right=0,
        )

    def gas_density(self, height):
        gas_density_filtered = np.where(
            self.gas_fraction_array <= 0, np.NaN, self.gas_density_array
        )
        return np.interp(height, self.height_array, gas_density_filtered, left=np.NaN)