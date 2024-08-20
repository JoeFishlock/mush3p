from pytest import approx
from mush3p.params import PhysicalParams


def test_concentration_ratio() -> None:
    far_salinity: float = 35.0
    liquidus_slope: float = 0.07
    eutectic_temperature: float = -21.2
    eutectic_salinity = -eutectic_temperature / liquidus_slope
    concentration_ratio: float = far_salinity / (eutectic_salinity - far_salinity)
    assert PhysicalParams(
        name="test",
        far_salinity=far_salinity,
        eutectic_temperature=eutectic_temperature,
        liquidus_slope=liquidus_slope,
    ).concentration_ratio == approx(concentration_ratio)
