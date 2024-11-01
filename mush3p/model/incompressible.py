import numpy as np
from .full import FullModel


class IncompressibleModel(FullModel):
    """implement equations for the incompressible model.

    These are identical to the full model but with the dimensionless gas density set to 1.0.
    """

    @property
    def gas_density(
        self,
    ):
        return np.ones_like(self.temperature)
