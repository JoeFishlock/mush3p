from functools import partial
from scipy.integrate import solve_bvp

from .output import NonDimensionalResults
from .model import MODEL_OPTIONS
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
