import numpy as np
from numpy.typing import NDArray

# Tolerances for error checking
VOLUME_SUM_TOLERANCE = 1e-8

GAS_FRACTION_GUESS: float = 0.01

# From Maus paper
PORE_THROAT_SCALING: float = 0.5
DRAG_EXPONENT: int = 6

# Initial Conditions for solver
INITIAL_MESH_NODES: int = 20
INITIAL_HEIGHT: NDArray = np.linspace(-1, 0, INITIAL_MESH_NODES)
INITIAL_TEMPERATURE: NDArray = np.linspace(0, -1, INITIAL_MESH_NODES)
INITIAL_TEMPERATURE_DERIVATIVE = np.full_like(INITIAL_TEMPERATURE, -1.0)
INITIAL_DISSOLVED_GAS_CONCENTRATION = np.linspace(0.8, 1.0, INITIAL_MESH_NODES)
INITIAL_HYDROSTATIC_PRESSURE = np.linspace(-0.1, 0, INITIAL_MESH_NODES)
INITIAL_FROZEN_GAS_FRACTION = np.full_like(INITIAL_TEMPERATURE, 0.02)
INITIAL_MUSHY_LAYER_DEPTH = np.full_like(INITIAL_TEMPERATURE, 1.5)


def get_initial_solution(model_choice: str):
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
    else:
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
