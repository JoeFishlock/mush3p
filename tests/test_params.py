from pytest import approx
from mush3p.params import PhysicalParams


def test_concentration_ratio() -> None:
    far_salinity: float = 34.0
    eutectic_salinity: float = 230.0
    concentration_ratio: float = far_salinity / (eutectic_salinity - far_salinity)
    assert PhysicalParams(
        name="test", far_salinity=far_salinity, eutectic_salinity=eutectic_salinity
    ).concentration_ratio == approx(concentration_ratio)
