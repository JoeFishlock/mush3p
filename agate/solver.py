from scipy.integrate import solve_bvp
from agate.output import NonDimensionalResults
from agate.model import MODEL_OPTIONS


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


def solve(non_dimensional_params):
    if non_dimensional_params.model_choice not in MODEL_OPTIONS.keys():
        raise ValueError(
            f"model_choice must be one of the implemented: {MODEL_OPTIONS.keys()}"
        )

    model = non_dimensional_params.create_model()
    solution_object = solve_bvp(
        model.ode_fun,
        model.boundary_conditions,
        model.INITIAL_HEIGHT,
        model.INITIAL_VARIABLES,
        verbose=0,
    )
    if not solution_object.success:
        raise RuntimeError(
            f"Could not solve {non_dimensional_params.name}.\nSolver exited with:\n{solution_object.message}"
        )

    temperature_array = get_array_from_solution(solution_object, "temperature")
    temperature_derivative_array = get_array_from_solution(
        solution_object, "temperature_derivative"
    )

    if non_dimensional_params.model_choice == "instant":
        hydrostatic_pressure_array = solution_object.y[2]
        frozen_gas_fraction = solution_object.y[3][-1]
        mushy_layer_depth = solution_object.y[4][0]
    else:
        hydrostatic_pressure_array = get_array_from_solution(
            solution_object, "hydrostatic_pressure"
        )
        frozen_gas_fraction = get_array_from_solution(
            solution_object, "frozen_gas_fraction"
        )[-1]
        mushy_layer_depth = get_array_from_solution(
            solution_object, "mushy_layer_depth"
        )[0]

    height_array = solution_object.x

    # Need to make this distinction as instant model doesn't solve ode for concentration
    # TODO: refactor to remove this if statement <09-01-23, Joe Fishlock> #
    if non_dimensional_params.model_choice == "instant":
        solid_fraction = model.calculate_solid_fraction(temperature=temperature_array)
        liquid_fraction = model.calculate_liquid_fraction(solid_fraction=solid_fraction)
        concentration_array = model.calculate_dissolved_gas_concentration(
            liquid_fraction=liquid_fraction
        )
    else:
        concentration_array = get_array_from_solution(solution_object, "concentration")

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