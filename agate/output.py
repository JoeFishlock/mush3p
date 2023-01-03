"""Class for storing simulation output"""
import numpy as np


def get_array_from_solution(solution_object, variable):
    variables = {
        "temperature": 0,
        "temperature_derivative": 1,
        "concentration": 2,
        "hydrostatic_pressure": 3,
        "frozen_gas_fraction": 4,
        "mushy_layer_depth": 5,
    }
    if variable not in variables.keys():
        raise ValueError(f"Invalid variable. Expected one of {variables.keys()}")

    return solution_object.y[variables[variable]]


class NonDimensionalResults:
    """class to store non-dimensional results of a simulation"""

    def __init__(self, non_dimensional_parameters):
        self.name = non_dimensional_parameters.name
        solution_object = non_dimensional_parameters.solve()
        self.params = non_dimensional_parameters
        self.temperature_array = get_array_from_solution(solution_object, "temperature")
        self.temperature_derivative_array = get_array_from_solution(
            solution_object, "temperature_derivative"
        )
        self.concentration_array = get_array_from_solution(
            solution_object, "concentration"
        )
        self.hydrostatic_pressure_array = get_array_from_solution(
            solution_object, "hydrostatic_pressure"
        )
        self.frozen_gas_fraction = get_array_from_solution(
            solution_object, "frozen_gas_fraction"
        )[-1]
        self.mushy_layer_depth = get_array_from_solution(
            solution_object, "mushy_layer_depth"
        )[0]
        self.height_array = solution_object.x

        # Calculate all mushy layer arrays
        # TODO: Add setter method to filter gas_density <03-01-23, Joe Fishlock> #
        model = non_dimensional_parameters.create_model()
        (
            self.solid_salinity_array,
            self.liquid_salinity_array,
            self.solid_fraction_array,
            self.liquid_fraction_array,
            self.gas_fraction_array,
            self.gas_density_array,
            self.liquid_darcy_velocity_array,
            self.gas_darcy_velocity_array,
        ) = model.calculate_all_variables(
            temperature=self.temperature_array,
            temperature_derivative=self.temperature_derivative_array,
            dissolved_gas_concentration=self.concentration_array,
            hydrostatic_pressure=self.hydrostatic_pressure_array,
            frozen_gas_fraction=self.frozen_gas_fraction,
            mushy_layer_depth=self.mushy_layer_depth,
            height=self.height_array,
        )

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
            height, self.height_array, self.liquid_darcy_velocity_array, right=0
        )

    def gas_darcy_velocity(self, height):
        return np.interp(
            height, self.height_array, self.gas_darcy_velocity_array, left=0, right=0
        )

    def gas_density(self, height):
        gas_density_filtered = np.where(
            self.gas_fraction_array <= 0, np.NaN, self.gas_density_array
        )
        return np.interp(height, self.height_array, gas_density_filtered, left=np.NaN)
