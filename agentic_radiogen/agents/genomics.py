from __future__ import annotations

from agentic_radiogen.schemas.contracts import GenomicMatrix, OmicsBundle


class GenomicsAgent:
    """Consumes an OmicsBundle only. Never calls the imaging agent."""

    def extract(self, bundle: OmicsBundle) -> GenomicMatrix:
        if not bundle.patient_ids:
            raise ValueError("OmicsBundle has no patients")
        features: dict[str, dict[str, float]] = {}
        names: set[str] = set()
        for pid in bundle.patient_ids:
            row: dict[str, float] = {}
            for gene, flag in bundle.mutations.get(pid, {}).items():
                row[f"{gene}_mut"] = float(flag)
            for gene, value in bundle.expression.get(pid, {}).items():
                row[f"{gene}_expr"] = float(value)
            clinical = bundle.clinical.get(pid, {})
            if "OS_time" in clinical:
                row["OS_time"] = float(clinical["OS_time"])
            if "OS_event" in clinical:
                row["OS_event"] = float(clinical["OS_event"])
            mut_values = [v for k, v in row.items() if k.endswith("_mut")]
            if mut_values:
                row["TMB_proxy"] = float(sum(mut_values))
            if not row:
                raise ValueError(f"No genomic payload for patient {pid}")
            features[pid] = row
            names.update(row)
        return GenomicMatrix(
            patient_ids=list(bundle.patient_ids),
            feature_names=sorted(names),
            features=features,
        )
