from __future__ import annotations

from typing import Any

from agentic_radiogen.schemas.contracts import GenomicMatrix, OmicsBundle

# Common clinical / literature aliases → HGNC-like symbols used in GDC.
_GENE_ALIASES: dict[str, str] = {
    "HER2": "ERBB2",
    "HER-2": "ERBB2",
    "NEU": "ERBB2",
    "C-ERBB-2": "ERBB2",
    "P53": "TP53",
    "P16": "CDKN2A",
    "P16INK4A": "CDKN2A",
    "ALK1": "ALK",
    "C-MET": "MET",
    "CMET": "MET",
    "PD-L1": "CD274",
    "PDL1": "CD274",
}

# Soft family groups: literature gene can relate to these cohort genes.
_GENE_FAMILIES: dict[str, frozenset[str]] = {
    "EGFR": frozenset({"EGFR", "ERBB2", "ERBB3", "ERBB4"}),
    "ERBB2": frozenset({"EGFR", "ERBB2", "ERBB3", "ERBB4"}),
    "ERBB3": frozenset({"EGFR", "ERBB2", "ERBB3", "ERBB4"}),
    "ERBB4": frozenset({"EGFR", "ERBB2", "ERBB3", "ERBB4"}),
    "KRAS": frozenset({"KRAS", "NRAS", "HRAS"}),
    "NRAS": frozenset({"KRAS", "NRAS", "HRAS"}),
    "HRAS": frozenset({"KRAS", "NRAS", "HRAS"}),
    "TP53": frozenset({"TP53", "TP63", "TP73"}),
    "BRCA1": frozenset({"BRCA1", "BRCA2"}),
    "BRCA2": frozenset({"BRCA1", "BRCA2"}),
    "CDKN2A": frozenset({"CDKN2A", "CDKN2B"}),
    "CDKN2B": frozenset({"CDKN2A", "CDKN2B"}),
}

# Clinical covariates for --clinic_data (predictors). Survival OS_* stay
# outcomes-only and are always available for C-index when present.
_CLINIC_NUMERIC = ("age_at_diagnosis",)
_CLINIC_CATEGORICAL = (
    "stage",
    "subtype",
    "primary_diagnosis",
    "vital_status",
    "sex",
    "gender",
    "race",
    "ethnicity",
)


class GenomicsAgent:
    """Consumes an OmicsBundle only. Never calls the imaging agent.

    Also maps cohort gene symbols ↔ literature gene claims (aliases / families).
    """

    def __init__(self, *, include_clinical: bool = False) -> None:
        self.include_clinical = bool(include_clinical)

    def extract(self, bundle: OmicsBundle) -> GenomicMatrix:
        if not bundle.patient_ids:
            raise ValueError("OmicsBundle has no patients")
        features: dict[str, dict[str, float]] = {}
        names: set[str] = set()
        clinic_by_pid = {
            pid: dict(bundle.clinical.get(pid, {}) or {}) for pid in bundle.patient_ids
        }
        clinic_cols = (
            _encode_clinical_matrix(clinic_by_pid) if self.include_clinical else {}
        )
        for pid in bundle.patient_ids:
            row: dict[str, float] = {}
            for gene, flag in bundle.mutations.get(pid, {}).items():
                row[f"{gene}_mut"] = float(flag)
            for gene, value in bundle.expression.get(pid, {}).items():
                row[f"{gene}_expr"] = float(value)
            clinical = clinic_by_pid.get(pid, {})
            if "OS_time" in clinical:
                row["OS_time"] = float(clinical["OS_time"])
            if "OS_event" in clinical:
                row["OS_event"] = float(clinical["OS_event"])
            if self.include_clinical:
                row.update(clinic_cols.get(pid, {}))
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

    @staticmethod
    def gene_symbol(genomic_feature: str) -> str:
        """Strip _mut/_expr suffix and normalize aliases."""
        gene = genomic_feature
        if gene.endswith("_mut") or gene.endswith("_expr"):
            gene = gene.rsplit("_", 1)[0]
        return GenomicsAgent.normalize_gene(gene)

    @staticmethod
    def normalize_gene(gene: str) -> str:
        g = (gene or "").strip().upper().replace("HER2", "ERBB2")
        return _GENE_ALIASES.get(g, g)

    def relate_to_literature_gene(
        self, genomic_feature: str, literature_gene: str
    ) -> tuple[bool, str, float]:
        """Return (related?, explanation, score) vs a literature gene symbol."""
        ours = self.gene_symbol(genomic_feature)
        lit = self.normalize_gene(literature_gene)
        if not ours or not lit:
            return False, "missing gene symbol", 0.0
        if ours == lit:
            return True, f"exact gene match ({ours})", 1.0
        # Alias already normalized above; keep explicit message if raw differed
        raw_ours = genomic_feature.rsplit("_", 1)[0].upper() if genomic_feature else ""
        if self.normalize_gene(raw_ours) == lit and raw_ours != lit:
            return True, f"alias {raw_ours} → {lit}", 0.95

        family = _GENE_FAMILIES.get(lit) or _GENE_FAMILIES.get(ours)
        if family and ours in family and lit in family:
            return (
                True,
                f"gene-family relation ({ours} <-> {lit} in {sorted(family)})",
                0.7,
            )
        return False, f"no relation ({ours} vs {lit})", 0.0

    def related_literature_genes(
        self, genomic_feature: str, literature_genes: list[str] | set[str]
    ) -> list[tuple[str, str, float]]:
        """List (lit_gene, reason, score) for literature genes related to ours."""
        out: list[tuple[str, str, float]] = []
        for lit in literature_genes:
            ok, reason, score = self.relate_to_literature_gene(genomic_feature, lit)
            if ok:
                out.append((self.normalize_gene(lit), reason, score))
        out.sort(key=lambda x: -x[2])
        return out


def _encode_clinical_matrix(
    clinic_by_pid: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Build per-patient clinic_* predictors (numeric + cohort label codes)."""
    out: dict[str, dict[str, float]] = {pid: {} for pid in clinic_by_pid}
    for key in _CLINIC_NUMERIC:
        for pid, clinical in clinic_by_pid.items():
            raw = clinical.get(key)
            if raw in (None, ""):
                continue
            try:
                out[pid][f"clinic_{key}"] = float(raw)
            except (TypeError, ValueError):
                continue
    for key in _CLINIC_CATEGORICAL:
        values = {
            pid: str(clinical.get(key) or "").strip()
            for pid, clinical in clinic_by_pid.items()
        }
        levels = sorted({v for v in values.values() if v})
        if not levels:
            continue
        code = {level: float(i + 1) for i, level in enumerate(levels)}
        for pid, raw in values.items():
            if raw:
                out[pid][f"clinic_{key}"] = code[raw]
    return out
