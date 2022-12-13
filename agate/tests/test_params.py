from agate.params import FullPhysicalParams
from pytest import approx


def test_concentration_ratio() -> None:
    far_salinity: float = 34.0
    eutectic_salinity: float = 230.0
    concentration_ratio: float = far_salinity / (eutectic_salinity - far_salinity)
    assert FullPhysicalParams(
        name="test", far_salinity=far_salinity, eutectic_salinity=eutectic_salinity
    ).concentration_ratio == approx(concentration_ratio)
