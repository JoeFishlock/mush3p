from functools import partial
import numpy as np
from scipy.integrate import solve_bvp
from agate.output import NonDimensionalResults
from agate.model import MODEL_OPTIONS
from scipy.integrate import simpson
from .static_settings import get_initial_solution, INITIAL_HEIGHT
from .boundary_conditions import get_boundary_conditions


def ode_fun(non_dimensional_params, height, variables):
    model_instance = MODEL_OPTIONS[non_dimensional_params.model_choice](
        non_dimensional_params, height, *variables
    )
    return model_instance.ode_fun


def solve(non_dimensional_params, max_nodes=1000):
    if non_dimensional_params.model_choice not in MODEL_OPTIONS.keys():
        raise ValueError(
            f"model_choice must be one of the implemented: {MODEL_OPTIONS.keys()}"
        )

    solution_object = solve_bvp(
        partial(ode_fun, non_dimensional_params),
        partial(get_boundary_conditions, non_dimensional_params),
        INITIAL_HEIGHT,
        get_initial_solution(non_dimensional_params.model_choice),
        max_nodes=max_nodes,
        verbose=0,
    )
    if not solution_object.success:
        raise RuntimeError(
            f"Could not solve {non_dimensional_params.name}.\nSolver exited with:\n{solution_object.message}"
        )

    height_array = solution_object.x
    temperature_array = solution_object.y[0]
    temperature_derivative_array = solution_object.y[1]

    if non_dimensional_params.model_choice == "instant":
        hydrostatic_pressure_array = solution_object.y[2]
        frozen_gas_fraction = solution_object.y[3][-1]
        mushy_layer_depth = solution_object.y[4][0]

        concentration_array = MODEL_OPTIONS[non_dimensional_params.model_choice](
            non_dimensional_params,
            height_array,
            temperature_array,
            temperature_derivative_array,
            hydrostatic_pressure_array,
            frozen_gas_fraction,
            mushy_layer_depth,
        ).dissolved_gas_concentration
    else:
        concentration_array = solution_object.y[2]
        hydrostatic_pressure_array = solution_object.y[3]
        frozen_gas_fraction = solution_object.y[4][-1]
        mushy_layer_depth = solution_object.y[5][0]

    return NonDimensionalResults(
        non_dimensional_parameters=non_dimensional_params,
        temperature_array=temperature_array,
        temperature_derivative_array=temperature_derivative_array,
        concentration_array=concentration_array,
        hydrostatic_pressure_array=hydrostatic_pressure_array,
        frozen_gas_fraction=frozen_gas_fraction,
        mushy_layer_depth=mushy_layer_depth,
        height_array=height_array,
    )


def calculate_RMSE(target_array, true_array, target_positions, true_positions):
    target = np.interp(true_positions, target_positions, target_array)
    diff = (target - true_array) ** 2
    normal = true_array**2
    numerator = simpson(diff, true_positions)
    denominator = simpson(normal, true_positions)
    """To test this function test easy one"""
    # x = np.linspace(0, 1, 100)
    # on = np.ones_like(x)
    # calculate_RMSE(x+1, on, x, x)
    """Answer should be sqrt(1/3)"""
    return np.sqrt(numerator / denominator)


def compare_model_to_full(reduced_model_results: NonDimensionalResults):
    parameters = reduced_model_results.params
    parameters.model_choice = "full"
    base_results = solve(parameters)
    print(
        calculate_RMSE(
            reduced_model_results.gas_fraction_array,
            base_results.gas_fraction_array,
            reduced_model_results.height_array,
            base_results.height_array,
        )
    )
