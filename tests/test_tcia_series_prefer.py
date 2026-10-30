from agentic_radiogen.data.tcia_client import (
    TciaSeries,
    has_positive_chest_evidence,
    is_non_chest_series,
    prefer_volumetric_series,
    prefer_volumetric_series_ranked,
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


def test_chest_only_prefers_labeled_chest_over_thicker_empty_desc() -> None:
    """Empty site_text must not beat a thinner labeled CHEST series (Z021-class)."""
    empty_thick = TciaSeries(
        "TCGA-17-Z021",
        "uid-empty",
        "CT",
        "TCGA-LUAD",
        image_count=400,
        series_description="",
        body_part_examined="CHEST",  # unreliable alone; must not whitelist
    )
    chest = TciaSeries(
        "TCGA-17-Z021",
        "uid-chest",
        "CT",
        "TCGA-LUAD",
        image_count=80,
        series_description="CHEST  5.0  B30f",
    )
    assert not has_positive_chest_evidence(empty_thick)
    assert has_positive_chest_evidence(chest)
    ranked = prefer_volumetric_series_ranked([empty_thick, chest], chest_only=True)
    assert [s.series_uid for s in ranked] == ["uid-chest", "uid-empty"]
    best = prefer_volumetric_series([empty_thick, chest], chest_only=True)
    assert best is not None and best.series_uid == "uid-chest"


def test_body_part_abdomen_blacklists_when_description_empty() -> None:
    abd_body = TciaSeries(
        "p",
        "uid-abd-body",
        "CT",
        "TCGA-LUAD",
        image_count=100,
        series_description="",
        body_part_examined="ABDOMEN",
    )
    assert is_non_chest_series(abd_body)
    assert prefer_volumetric_series([abd_body], chest_only=True) is None


def test_ranked_returns_multiple_chest_candidates() -> None:
    thin_chest = TciaSeries(
        "p", "uid-thin", "CT", "c", image_count=40, series_description="CHEST"
    )
    thick_chest = TciaSeries(
        "p", "uid-thick", "CT", "c", image_count=120, series_description="LUNG"
    )
    abd = TciaSeries(
        "p", "uid-abd", "CT", "c", image_count=200, series_description="ABD/PEL"
    )
    ranked = prefer_volumetric_series_ranked(
        [abd, thin_chest, thick_chest], chest_only=True
    )
    assert [s.series_uid for s in ranked] == ["uid-thick", "uid-thin"]


def test_patient_series_lookup_preserves_full_metadata_without_modality_filter() -> None:
    calls: list[dict[str, str]] = []

    def getter(_url: str, params: dict[str, str]) -> list[dict[str, object]]:
        calls.append(params)
        return [
            {
                "PatientID": "TCGA-17-Z021",
                "SeriesInstanceUID": "uid-ct",
                "Modality": "CT",
                "Collection": "TCGA-LUAD",
                "StudyInstanceUID": "study-uid",
                "StudyDate": "20010101",
                "SeriesNumber": 3,
                "ImageCount": 49,
            }
        ]

    from agentic_radiogen.data.tcia_client import TciaClient

    series = TciaClient(getter=getter).list_series_for_patient("TCGA-17-Z021")

    assert calls == [{"PatientID": "TCGA-17-Z021"}]
    assert len(series) == 1
    assert series[0].modality == "CT"
    assert series[0].source_metadata is not None
    assert series[0].source_metadata["StudyInstanceUID"] == "study-uid"
