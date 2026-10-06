from __future__ import annotations

from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.util.progress import log, progress_iter


def test_progress_iter_yields_all_items(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RADIOGEN_QUIET", "0")
    items = list(progress_iter([1, 2, 3], desc="test", unit="x", enabled=True))
    assert items == [1, 2, 3]


def test_log_can_be_silenced(monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENTIC_RADIOGEN_QUIET", "1")
    log("should not appear")
    assert capsys.readouterr().err == ""


def test_live_catalog_records_intermediate(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_RADIOGEN_QUIET", "1")

    class FakeTcia:
        def list_series(self, collection, modality):
            from agentic_radiogen.data.tcia_client import TciaSeries

            assert collection == "CPTAC-PDA"
            return [
                TciaSeries("C3L-1", "1.2.3", modality, collection),
                TciaSeries("C3L-2", "1.2.4", modality, collection),
            ]

        def download_series(self, series_uid):
            raise AssertionError("download should be skipped")

    class FakeGdc:
        def list_cases(self, project_id, submitter_ids=None, size=2000, page_size=1000):
            from agentic_radiogen.data.gdc_client import GdcCase

            assert project_id == "CPTAC-3"
            cases = [
                GdcCase(submitter_id="C3L-1", clinical={}),
                GdcCase(submitter_id="C3L-ONLY-GDC", clinical={}),
            ]
            if submitter_ids is not None:
                wanted = set(submitter_ids)
                cases = [c for c in cases if c.submitter_id in wanted]
            return cases

        def mutation_flags(self, submitter_ids, genes):
            return {pid: {g: 0 for g in genes} for pid in submitter_ids}

        def expression_means(self, submitter_ids, genes):
            return {pid: {g: 0.0 for g in genes} for pid in submitter_ids}

    catalog = LiveCatalog(
        tcia=FakeTcia(),
        gdc=FakeGdc(),
        download_dicom=False,
        extract_radiomics=False,
        show_progress=False,
    )
    rows = catalog.query_metadata(disease="pancreas", modality="CT", genes=["KRAS"])
    assert len(rows) == 1
    assert catalog.last_intermediate["tcia_series"] == 2
    assert catalog.last_intermediate["tcia_patients"] == 2
    assert catalog.last_intermediate["gdc_patients"] == 2
    assert catalog.last_intermediate["paired_available"] == 1
    assert catalog.last_intermediate["tcia_collection"] == "CPTAC-PDA"
    assert catalog.last_intermediate["gdc_project"] == "CPTAC-3"
