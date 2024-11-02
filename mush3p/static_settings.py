import numpy as np
from numpy.typing import NDArray

# Tolerances for error checking
VOLUME_SUM_TOLERANCE = 1e-8

# Wall drag enhancement factor 1/K1 from hartholt et al 1994
def HARTHOLT_DRAG_FUNCTION(L):
    return (1 - 1.4567 * L + 1.4567 * L**5 - L**6) / (1 + 1.4567 * L**5)


# Initial Conditions for solver
INITIAL_MESH_NODES: int = 20
INITIAL_HEIGHT = np.linspace(-1, 0, INITIAL_MESH_NODES)
INITIAL_TEMPERATURE = np.linspace(0, -1, INITIAL_MESH_NODES)
INITIAL_TEMPERATURE_DERIVATIVE = np.full_like(INITIAL_TEMPERATURE, -1.0)
INITIAL_DISSOLVED_GAS_CONCENTRATION = np.linspace(0.8, 1.0, INITIAL_MESH_NODES)
INITIAL_HYDROSTATIC_PRESSURE = np.linspace(-0.1, 0, INITIAL_MESH_NODES)
INITIAL_FROZEN_GAS_FRACTION = np.full_like(INITIAL_TEMPERATURE, 0.02)
INITIAL_MUSHY_LAYER_DEPTH = np.full_like(INITIAL_TEMPERATURE, 1.5)


def get_initial_solution(model_choice: str) -> NDArray:
    """Get intial solution for scipy solve_BVP which satisfies the boundary conditions.

    Args:
        model_choice (str): one of "full", "incompressible" "reduced" or "instant"
    Returns:
        NDArray: initial solution for scipy solve_BVP
    """
    if model_choice == "instant":
        return np.vstack(
            (
                INITIAL_TEMPERATURE,
                INITIAL_TEMPERATURE_DERIVATIVE,
                INITIAL_HYDROSTATIC_PRESSURE,
                INITIAL_FROZEN_GAS_FRACTION,
                INITIAL_MUSHY_LAYER_DEPTH,
            )
        )
    return np.vstack(
        (
            INITIAL_TEMPERATURE,
            INITIAL_TEMPERATURE_DERIVATIVE,
            INITIAL_DISSOLVED_GAS_CONCENTRATION,
            INITIAL_HYDROSTATIC_PRESSURE,
            INITIAL_FROZEN_GAS_FRACTION,
            INITIAL_MUSHY_LAYER_DEPTH,
        )
    )
