"""Class for storing simulation output"""


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
        solution_object = non_dimensional_parameters.solve()
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
        )
        self.mushy_layer_depth = get_array_from_solution(
            solution_object, "mushy_layer_depth"
        )
        self.height_array = solution_object.x
