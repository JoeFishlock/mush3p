"""
The equations for solving the incompressible model

All quantities are calculated from the smaller set of variables:
temperature
temperature_derivative
dissolved_gas_concentration
hydrostatic_pressure
frozen_gas_fraction
mushy_layer_depth

height (vertical coordinate)
"""

import numpy as np
from .full import FullModel


class IncompressibleModel(FullModel):
    """Class containing equations with no gas compressibility.
    The non dimensional gas density is set to 1.0"""

    @property
    def gas_density(
        self,
    ):
        return np.ones_like(self.temperature)
