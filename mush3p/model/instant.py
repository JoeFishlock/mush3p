"""
The equations for solving the instant nucleation model

All quantities are calculated from the smaller set of variables:
temperature
temperature_derivative
hydrostatic_pressure
frozen_gas_fraction
mushy_layer_depth

height (vertical coordinate)
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from .reduced import ReducedModel
from ..params import NonDimensionalParams


class InstantNucleationModel(ReducedModel):
    """Implements the equations to solve the instant model.

    This is an extension of the reduced model where the nucleation rate is infinite and
    and so any supersaturations is immediately converted to the gas phase.
    This means we do not need to solve the ODE for dissolved gas concentration in this
    case.

    Args:
        params (NonDimensionalParams): Non-dimensional parameters
        height (NDArray): Height values
        temperature (NDArray): Temperature values
        temperature_derivative (NDArray): Temperature derivative values
        hydrostatic_pressure (NDArray): Hydrostatic pressure values
        frozen_gas_fraction (NDArray): Frozen gas fraction
        mushy_layer_depth (NDArray): Mushy layer depth"""

    def __init__(
        self,
        params: NonDimensionalParams,
        height: NDArray,
        temperature: NDArray,
        temperature_derivative: NDArray,
        hydrostatic_pressure: NDArray,
        frozen_gas_fraction: NDArray,
        mushy_layer_depth: NDArray,
    ) -> None:
        self.params = params
        self.height = height
        self.temperature = temperature
        self.temperature_derivative = temperature_derivative
        self.hydrostatic_pressure = hydrostatic_pressure
        self.frozen_gas_fraction = frozen_gas_fraction
        self.mushy_layer_depth = mushy_layer_depth

    @property
    def dissolved_gas_concentration(self):
        return np.minimum(
            np.ones_like(self.liquid_fraction),
            self.params.far_dissolved_concentration_scaled / self.liquid_fraction,
        )

    @property
    def ode_fun(self) -> Any:

        if not self.check_volume_fractions_sum_to_one:
            raise ValueError("Volume fractions do not sum to 1")

        return np.vstack(
            (
                self.temperature_derivative,
                self.temperature_second_derivative,
                self.hydrostatic_pressure_derivative,
                np.zeros_like(self.temperature),
                np.zeros_like(self.temperature),
            )
        )
