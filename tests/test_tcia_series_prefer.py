from agentic_radiogen.data.tcia_client import TciaSeries, prefer_volumetric_series


def test_prefer_volumetric_over_scout() -> None:
    scout = TciaSeries(
        "TCGA-01-0001",
        "uid-scout",
        "CT",
        "TCGA-LUSC",
        image_count=2,
        series_description="Scout",
    )
    volume = TciaSeries(
        "TCGA-01-0001",
        "uid-vol",
        "CT",
        "TCGA-LUSC",
        image_count=120,
        series_description="CHEST CT",
    )
    thin = TciaSeries(
        "TCGA-01-0001",
        "uid-thin",
        "CT",
        "TCGA-LUSC",
        image_count=5,
        series_description="AXIAL",
    )
    best = prefer_volumetric_series([scout, thin, volume])
    assert best is not None
    assert best.series_uid == "uid-vol"


def test_prefer_thickest_when_no_description() -> None:
    a = TciaSeries("p", "a", "CT", "c", image_count=2)
    b = TciaSeries("p", "b", "CT", "c", image_count=80)
    assert prefer_volumetric_series([a, b]).series_uid == "b"  # type: ignore[union-attr]


def test_prefer_skips_localizer_even_if_listed_first() -> None:
    first = TciaSeries(
        "p", "uid1", "CT", "c", image_count=1, series_description="Localizer"
    )
    second = TciaSeries(
        "p", "uid2", "CT", "c", image_count=160, series_description="Axial Chest"
    )
    # Mimic old API order: thin first.
    best = prefer_volumetric_series([first, second])
    assert best is not None and best.series_uid == "uid2"
