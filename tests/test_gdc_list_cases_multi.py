from __future__ import annotations

from agentic_radiogen.data.gdc_client import GdcClient


def test_list_cases_multi_fetches_each_project_fully() -> None:
    """Pairable TCGA projects must not be truncated by a shared case budget."""
    calls: list[list[str]] = []

    def poster(_url: str, payload: dict) -> dict:
        projects = list(
            (((payload.get("filters") or {}).get("content") or {}).get("value") or [])
        )
        calls.append(projects)
        assert len(projects) == 1
        pid = projects[0]
        # Simulate a large first project that would formerly exhaust a shared 10k cap.
        if pid == "BIG":
            offset = int(payload.get("from") or 0)
            size = int(payload.get("size") or 1000)
            total = 2500
            hits = [
                {"submitter_id": f"BIG-{i:04d}", "diagnoses": []}
                for i in range(offset, min(offset + size, total))
            ]
            return {
                "data": {
                    "hits": hits,
                    "pagination": {"from": offset, "size": size, "total": total},
                }
            }
        if pid == "TCGA-LUAD":
            return {
                "data": {
                    "hits": [
                        {"submitter_id": "TCGA-05-4244", "diagnoses": []},
                        {"submitter_id": "TCGA-05-4249", "diagnoses": []},
                    ],
                    "pagination": {"total": 2},
                }
            }
        return {"data": {"hits": [], "pagination": {"total": 0}}}

    client = GdcClient(poster=poster)
    cases = client.list_cases_multi(["BIG", "TCGA-LUAD"], size=3000, page_size=1000)
    ids = {c.submitter_id for c in cases}
    assert calls[0] == ["BIG"]
    assert ["TCGA-LUAD"] in calls
    assert "TCGA-05-4244" in ids
    assert "TCGA-05-4249" in ids
    assert sum(1 for i in ids if i.startswith("BIG-")) == 2500
