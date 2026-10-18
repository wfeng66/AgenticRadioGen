from agentic_radiogen.data.tcia_client import (
    TciaSeries,
    is_non_chest_series,
    prefer_volumetric_series,
)


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


def test_chest_only_drops_abd_pel_over_chest() -> None:
    """TCGA-92-8065-style: ABD/PEL is thicker but must lose to CHEST."""
    abd = TciaSeries(
        "TCGA-92-8065",
        "uid-abd",
        "CT",
        "TCGA-LUSC",
        image_count=238,
        series_description="ABD/PEL  3.0  B20f",
    )
    chest = TciaSeries(
        "TCGA-92-8065",
        "uid-chest",
        "CT",
        "TCGA-LUSC",
        image_count=164,
        series_description="CHEST ROUTINE  3.0  B20f",
    )
    assert is_non_chest_series(abd)
    assert not is_non_chest_series(chest)
    # Without filter, abd wins on ImageCount.
    assert prefer_volumetric_series([abd, chest]).series_uid == "uid-abd"  # type: ignore[union-attr]
    best = prefer_volumetric_series([abd, chest], chest_only=True)
    assert best is not None and best.series_uid == "uid-chest"


def test_chest_only_drops_whole_body() -> None:
    """TCGA-38-6178-style: CT WB thicker than dedicated chest."""
    wb = TciaSeries(
        "TCGA-38-6178",
        "uid-wb",
        "CT",
        "TCGA-LUAD",
        image_count=329,
        series_description="CT WB  5.0  eFoV",
    )
    chest = TciaSeries(
        "TCGA-38-6178",
        "uid-chest",
        "CT",
        "TCGA-LUAD",
        image_count=62,
        series_description="Chest  5.0  B31f",
    )
    pe = TciaSeries(
        "TCGA-38-6178",
        "uid-pe",
        "CT",
        "TCGA-LUAD",
        image_count=229,
        series_description="PE STUDY ST",
    )
    assert is_non_chest_series(wb)
    best = prefer_volumetric_series([wb, chest, pe], chest_only=True)
    assert best is not None and best.series_uid == "uid-pe"


def test_chest_only_drops_patient_when_only_non_chest() -> None:
    only_abd = TciaSeries(
        "p",
        "uid-abd",
        "CT",
        "TCGA-LUSC",
        image_count=100,
        series_description="ABD/PEL  3.0  B20f",
    )
    assert prefer_volumetric_series([only_abd], chest_only=True) is None


def test_chest_abd_combined_is_eligible() -> None:
    """TCGA-92-7340-style: Chest/Abd must not be blacklisted as pure abdomen."""
    wb = TciaSeries(
        "TCGA-92-7340",
        "uid-wb",
        "CT",
        "TCGA-LUSC",
        image_count=449,
        series_description="CT WB 5.0 B31f",
    )
    chest_abd = TciaSeries(
        "TCGA-92-7340",
        "uid-chest-abd",
        "CT",
        "TCGA-LUSC",
        image_count=224,
        series_description="Chest/Abd  3.0  B20f",
    )
    topogram = TciaSeries(
        "TCGA-92-7340",
        "uid-topo",
        "CT",
        "TCGA-LUSC",
        image_count=1,
        series_description="Topogram  0.6  T20s",
    )
    assert is_non_chest_series(wb)
    assert not is_non_chest_series(chest_abd)
    best = prefer_volumetric_series([wb, chest_abd, topogram], chest_only=True)
    assert best is not None and best.series_uid == "uid-chest-abd"


def test_chest_only_drops_patient_when_only_topogram_left() -> None:
    """TCGA-17-Z062-style: Abd + Topogram → omit patient (no usable chest CT)."""
    abd = TciaSeries(
        "TCGA-17-Z062",
        "uid-abd",
        "CT",
        "TCGA-LUAD",
        image_count=356,
        series_description="Abd.CT 5.0 B30s",
    )
    topo = TciaSeries(
        "TCGA-17-Z062",
        "uid-topo",
        "CT",
        "TCGA-LUAD",
        image_count=1,
        series_description="Topogram",
    )
    assert prefer_volumetric_series([abd, topo], chest_only=True) is None
