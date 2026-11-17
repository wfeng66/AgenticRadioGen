"""--organ-fraction: try alternate series, then replace patient."""

from __future__ import annotations

from pathlib import Path

from agentic_radiogen.data.catalog import CatalogRecord
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.data.tcia_client import TciaSeries


class _FakeTcia:
    def __init__(self, fracs: dict[str, float], cache: Path) -> None:
        self.fracs = fracs
        self.cache_dir = cache
        self.downloads: list[str] = []

    def download_series(self, series_uid: str) -> Path:
        self.downloads.append(series_uid)
        dest = self.cache_dir / series_uid
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "dummy.dcm").write_text("x", encoding="utf-8")
        (dest / ".complete").write_text("ok", encoding="utf-8")
        return dest


def _rec(pid: str, uid: str, *, desc: str, body: str = "LUNG") -> CatalogRecord:
    return CatalogRecord(
        patient_id=pid,
        disease="lung",
        modality="CT",
        mutations={"TP53": 0},
        expression={},
        clinical={
            "_tcia_body_part": body,
            "_tcia_series_description": desc,
            "_tcia_collection": "TCGA-LUAD",
            "_tcia_image_count": 100,
            "_gdc_project": "TCGA-LUAD",
            "_tcia_series_candidates": [],
        },
        series_uid=uid,
        radiomic_features={},
        volume_summary={},
    )


def test_switches_to_alternate_series(tmp_path, monkeypatch) -> None:
    fracs = {"uid-bad": 0.22, "uid-good": 0.55}
    tcia = _FakeTcia(fracs, tmp_path / "tcia")

    def fake_qc(series_dir, *, organ_fraction=0.3):
        uid = Path(series_dir).name
        frac = fracs[uid]
        return frac >= organ_fraction, frac

    monkeypatch.setattr(
        "agentic_radiogen.imaging.chest_qc.series_meets_organ_fraction",
        fake_qc,
    )

    catalog = LiveCatalog(tcia=tcia, download_dicom=True, extract_radiomics=False)
    catalog.last_intermediate = {
        "disease_query": "lung adenocarcinoma",
        "organ_fraction": 0.5,
    }
    catalog._record_cache["P1"] = _rec("P1", "uid-bad", desc="LUNG WINDOWS")
    catalog._record_cache["P1"].clinical["_tcia_series_candidates"] = [
        "uid-bad",
        "uid-good",
    ]
    catalog.gdc.mutation_flags = lambda ids, genes: {i: {} for i in ids}  # type: ignore[method-assign]
    catalog.gdc.expression_means = lambda ids, genes: {i: {} for i in ids}  # type: ignore[method-assign]

    out = catalog.fetch_records(
        ["P1"], target_count=1, organ_fraction=0.5
    )
    assert len(out) == 1
    assert out[0].series_uid == "uid-good"
    assert tcia.downloads == ["uid-bad", "uid-good"]


def test_replaces_patient_when_all_series_fail(tmp_path, monkeypatch) -> None:
    fracs = {"uid-bad": 0.10, "uid-also-bad": 0.12, "uid-ok": 0.60}
    tcia = _FakeTcia(fracs, tmp_path / "tcia")

    def fake_qc(series_dir, *, organ_fraction=0.3):
        uid = Path(series_dir).name
        frac = fracs.get(uid, 0.0)
        return frac >= organ_fraction, frac

    monkeypatch.setattr(
        "agentic_radiogen.imaging.chest_qc.series_meets_organ_fraction",
        fake_qc,
    )

    catalog = LiveCatalog(tcia=tcia, download_dicom=True, extract_radiomics=False)
    catalog.last_intermediate = {
        "disease_query": "lung cancer",
        "organ_fraction": 0.5,
    }
    catalog._record_cache["P1"] = _rec("P1", "uid-bad", desc="LUNG WINDOWS")
    catalog._record_cache["P1"].clinical["_tcia_series_candidates"] = [
        "uid-bad",
        "uid-also-bad",
    ]
    catalog._record_cache["P2"] = _rec("P2", "uid-ok", desc="CHEST CT", body="CHEST")
    catalog._record_cache["P2"].clinical["_tcia_series_candidates"] = ["uid-ok"]
    catalog.gdc.mutation_flags = lambda ids, genes: {i: {} for i in ids}  # type: ignore[method-assign]
    catalog.gdc.expression_means = lambda ids, genes: {i: {} for i in ids}  # type: ignore[method-assign]

    out = catalog.fetch_records(
        ["P1", "P2"], target_count=1, organ_fraction=0.5
    )
    assert len(out) == 1
    assert out[0].patient_id == "P2"
    assert out[0].series_uid == "uid-ok"
