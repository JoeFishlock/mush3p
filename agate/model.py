from typing import Protocol
from numpy.typing import ArrayLike


class Model(Protocol):
    def equations(self) -> ArrayLike:
        ...
