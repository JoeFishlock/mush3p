from typing import Protocol
from agate.model import Model
from agate.output import NonDimensionalOutput


class NumericalParameters:
    tol: float = 1e-7


class NonDimensionalParams(Protocol):
    def solve(
        self, numerical_parameters: NumericalParameters, model: Model
    ) -> NonDimensionalOutput:
        ...


class PhysicalParams(Protocol):
    def non_dimensionalise(self) -> NonDimensionalParams:
        ...
