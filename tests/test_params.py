from mush3p.params import PhysicalParams


def test_calculating_concentration_ratio() -> None:
    far_salinity: float = 35.0
    liquidus_slope: float = 0.07
    eutectic_temperature: float = -21.2
    eutectic_salinity = -eutectic_temperature / liquidus_slope
    reference_concentration_ratio: float = far_salinity / (
        eutectic_salinity - far_salinity
    )

    computed_concentration_ratio = PhysicalParams(
        name="test",
        far_salinity=far_salinity,
        eutectic_temperature=eutectic_temperature,
        liquidus_slope=liquidus_slope,
    ).concentration_ratio

    assert computed_concentration_ratio == reference_concentration_ratio
