from typing import Protocol


class PhysicalOutput(Protocol):
    def save(self) -> None:
        ...


class NonDimensionalOutput(Protocol):
    def dimensionalise(self) -> PhysicalOutput:
        ...
