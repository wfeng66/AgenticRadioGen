from __future__ import annotations

from typing import Any

from agentic_radiogen.schemas.contracts import GenomicMatrix, ImageBundle, OmicsBundle, RadiomicMatrix


def build_patient_table_from_bundles(
    images: ImageBundle | None,
    omics: OmicsBundle | None,
) -> dict[str, Any]:
    """Fallback report rows from fetch bundles (precomputed radiomics + mutation flags)."""
    image_by_pid: dict[str, dict[str, float]] = {}
    if images:
        for series in images.series:
            row = {
                k: float(v)
                for k, v in (series.precomputed_features or {}).items()
                if not str(k).startswith("meta_")
            }
            if row:
                image_by_pid[series.patient_id] = row
    genes: list[str] = []
    if omics and omics.metadata.get("genes"):
        genes = list(omics.metadata["genes"])
    patient_ids = sorted(set(image_by_pid) | set(omics.patient_ids if omics else []))
    radio_names: set[str] = set()
    for row in image_by_pid.values():
        radio_names.update(row)
    genomic_feature_names = [f"{g}_mut" for g in genes] + [f"{g}_expr" for g in genes]
    patients: list[dict[str, Any]] = []
    for pid in patient_ids:
        radio = dict(image_by_pid.get(pid, {}))
        mut_map = omics.mutations.get(pid, {}) if omics else {}
        expr_map = omics.expression.get(pid, {}) if omics else {}
        mutations = {g: int(mut_map.get(g, 0)) for g in genes}
        expression = {g: float(expr_map.get(g, 0.0)) for g in genes}
        clinical = dict(omics.clinical.get(pid, {})) if omics else {}
        altered = [gene for gene, flag in mutations.items() if flag]
        patients.append(
            {
                "patient_id": pid,
                "radiomic_features": radio,
                "genomic_alterations": {
                    "mutations": mutations,
                    "altered_genes": altered,
                    "expression": expression,
                    "clinical": clinical,
                },
            }
        )
    return {
        "patient_ids": patient_ids,
        "n_patients": len(patient_ids),
        "radiomic_feature_names": sorted(radio_names),
        "genomic_feature_names": genomic_feature_names,
        "patients": patients,
        "source": "fetch_bundles",
    }


def resolve_patient_table(
    radiomics: RadiomicMatrix | None,
    genomics: GenomicMatrix | None,
    images: ImageBundle | None = None,
    omics: OmicsBundle | None = None,
) -> dict[str, Any]:
    table = build_patient_table(radiomics, genomics)
    if table["n_patients"] > 0:
        return table
    if images is not None or omics is not None:
        fallback = build_patient_table_from_bundles(images, omics)
        if fallback["n_patients"] > 0:
            return fallback
    return table


def build_patient_table(
    radiomics: RadiomicMatrix | None,
    genomics: GenomicMatrix | None,
) -> dict[str, Any]:
    """Separate per-patient radiomic features and genomic alterations for the report."""
    radio_ids = set(radiomics.patient_ids) if radiomics else set()
    geno_ids = set(genomics.patient_ids) if genomics else set()
    patient_ids = sorted(radio_ids & geno_ids) if radio_ids and geno_ids else sorted(radio_ids or geno_ids)

    patients: list[dict[str, Any]] = []
    for pid in patient_ids:
        radio = dict(radiomics.features.get(pid, {})) if radiomics else {}
        geno = dict(genomics.features.get(pid, {})) if genomics else {}
        mutations = {
            key.replace("_mut", ""): int(value)
            for key, value in geno.items()
            if key.endswith("_mut")
        }
        expression = {
            key.replace("_expr", ""): float(value)
            for key, value in geno.items()
            if key.endswith("_expr")
        }
        clinical = {
            key: float(value)
            for key, value in geno.items()
            if key in {"OS_time", "OS_event", "TMB_proxy"}
        }
        altered = [gene for gene, flag in mutations.items() if flag]
        patients.append(
            {
                "patient_id": pid,
                "radiomic_features": radio,
                "genomic_alterations": {
                    "mutations": mutations,
                    "altered_genes": altered,
                    "expression": expression,
                    "clinical": clinical,
                },
            }
        )

    return {
        "patient_ids": patient_ids,
        "n_patients": len(patient_ids),
        "radiomic_feature_names": list(radiomics.feature_names) if radiomics else [],
        "genomic_feature_names": list(genomics.feature_names) if genomics else [],
        "patients": patients,
    }


def mutation_prevalence(table: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count altered / wild-type patients per gene from patient_table."""
    counts: dict[str, dict[str, int]] = {}
    for row in table.get("patients") or []:
        mutations = (row.get("genomic_alterations") or {}).get("mutations") or {}
        for gene, flag in mutations.items():
            bucket = counts.setdefault(gene, {"altered": 0, "wildtype": 0})
            if int(flag):
                bucket["altered"] += 1
            else:
                bucket["wildtype"] += 1
    return counts


def associations_by_gene(
    associations: list[dict[str, Any]], *, top_per_gene: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Group association dicts by genomic_feature; keep best q-values per gene."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in associations:
        gene = str(item.get("genomic_feature") or "")
        grouped.setdefault(gene, []).append(item)
    out: dict[str, list[dict[str, Any]]] = {}
    for gene, items in grouped.items():
        ranked = sorted(items, key=lambda a: (a.get("q_value", 1.0), -abs(a.get("effect_size", 0.0))))
        out[gene] = ranked[:top_per_gene]
    return out


def print_patient_table(table: dict[str, Any], *, max_rows: int = 12) -> None:
    """Compact cohort view. Full per-patient radiomics stay in the JSON --out file."""
    print("-" * 60)
    print("Cohort: radiomics x genomic alterations")
    n = table.get("n_patients", 0)
    print(f"Paired patients: {n}")
    if table.get("radiomic_feature_names"):
        print(f"Radiomic features ({len(table['radiomic_feature_names'])}): "
              f"{', '.join(table['radiomic_feature_names'])}")
    if table.get("genomic_feature_names"):
        print(f"Genomic features : {', '.join(table['genomic_feature_names'])}")
    prevalence = mutation_prevalence(table)
    if prevalence:
        print("Mutation prevalence:")
        for gene in sorted(prevalence):
            alt = prevalence[gene]["altered"]
            wt = prevalence[gene]["wildtype"]
            print(f"  {gene}: {alt} altered / {alt + wt} ({100.0 * alt / max(1, alt + wt):.0f}%)")
    patients = table.get("patients") or []
    if not patients:
        print("  (no patients)")
        return
    print(f"Patient snapshot (first {min(max_rows, len(patients))} of {len(patients)}; "
          "full radiomics in --out JSON):")
    for row in patients[:max_rows]:
        geno = row.get("genomic_alterations") or {}
        altered = geno.get("altered_genes") or []
        mutations = geno.get("mutations") or {}
        flags = ", ".join(f"{g}={int(v)}" for g, v in mutations.items()) or "none"
        alt_txt = ", ".join(altered) if altered else "none"
        n_feat = len(row.get("radiomic_features") or {})
        print(f"  {row['patient_id']}: altered=[{alt_txt}]; {flags}; radiomics={n_feat} features")
    if len(patients) > max_rows:
        print(f"  ... {len(patients) - max_rows} more patients in JSON patient_table.patients")
