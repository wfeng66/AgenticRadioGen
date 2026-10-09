from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_radiogen.agents.data_matcher import DataMatcherAgent, DownloadDeniedError
from agentic_radiogen.agents.genomics import GenomicsAgent
from agentic_radiogen.agents.imaging import ImagingRadiomicsAgent
from agentic_radiogen.agents.literature import LiteratureAgent
from agentic_radiogen.agents.orchestrator import OrchestratorAgent
from agentic_radiogen.agents.statistics import StatisticalCriticalAgent
from agentic_radiogen.data.catalog import DemoCatalog
from agentic_radiogen.data.gate import AlwaysAllowGate, FlagGate
from agentic_radiogen.data.http_json import RemoteApiError
from agentic_radiogen.data.live_catalog import LiveCatalog
from agentic_radiogen.imaging.segment import segmentation_backend
from agentic_radiogen.pipeline.loop import DiscoveryLoop
from agentic_radiogen.pipeline.report_table import (
    associations_by_gene,
    build_patient_table,
    mutation_prevalence,
    print_patient_table,
    resolve_patient_table,
    write_associations_csv,
)
from agentic_radiogen.data.disease_match import format_dataset_extraction_note
from agentic_radiogen.pipeline.stage2 import extract_parallel
from agentic_radiogen.pipeline.stage3 import Stage3Error, join_and_interpret
from agentic_radiogen.schemas.contracts import ImageBundle, OmicsBundle


def print_summary(payload: dict[str, Any]) -> None:
    """Human-readable view of the result. Full JSON is optional via --json / --out."""
    stage = payload.get("stage")
    question = payload.get("question") or {}
    print("=" * 60)
    print(f"Stage {stage} summary")
    print("=" * 60)
    print(f"Disease : {question.get('disease')}")
    genes = question.get("genes") or []
    request = payload.get("request") or {}
    gene_source = (request.get("filters") or {}).get("gene_source")
    intermediate = payload.get("intermediate") or {}
    discovered = intermediate.get("genes") or request.get("genes") or []
    if gene_source == "literature" or request.get("genes"):
        panel = request.get("genes") or discovered
        print(f"Genes   : literature panel ({len(panel)}): {', '.join(panel[:12])}"
              f"{'...' if len(panel) > 12 else ''}")
    elif discovered:
        print(
            f"Genes   : cohort mutations ({len(discovered)}): {', '.join(discovered[:12])}"
            f"{'...' if len(discovered) > 12 else ''}"
        )
    else:
        print("Genes   : all mutations in paired genomics cohort (resolved at fetch)")
    print(f"Catalog : {payload.get('catalog')}")

    preview = payload.get("preview") or {}
    fetch = payload.get("fetch") or {}
    if preview:
        print(f"Paired  : {preview.get('n_paired')} patients (preview)")
    if fetch:
        print(f"Fetch   : {fetch.get('status')} ({fetch.get('n_images', '?')} image / "
              f"{fetch.get('n_omics', '?')} omics)")
        for note in (fetch.get("imaging_notes") or [])[:8]:
            print(f"  imaging: {note}")
    for note in (payload.get("imaging_notes") or [])[:8]:
        print(f"  imaging: {note}")
    intermediate = payload.get("intermediate") or {}
    if intermediate:
        print("-" * 60)
        print("Intermediate dataset / extraction progress")
        if intermediate.get("tcia_collection") or intermediate.get("gdc_project"):
            print(
                f"  Sources : TCIA {intermediate.get('tcia_collection')} "
                f"({intermediate.get('modality')}) ∩ GDC {intermediate.get('gdc_project')}"
            )
        if "tcia_series" in intermediate:
            print(
                f"  TCIA    : {intermediate.get('tcia_series')} series / "
                f"{intermediate.get('tcia_patients')} patients"
            )
        if "paired_patients" in intermediate or "paired_available" in intermediate:
            print(
                f"  Paired  : available={intermediate.get('paired_available', intermediate.get('paired_patients'))}, "
                f"selected={intermediate.get('paired_selected', '?')} "
                f"(max_patients={intermediate.get('max_patients', '?')}; "
                f"tcia-only dropped={intermediate.get('tcia_only_dropped', '?')}, "
                f"gdc-only dropped={intermediate.get('gdc_only_dropped', '?')})"
            )
        if "gdc_patients" in intermediate:
            print(f"  GDC project patients: {intermediate.get('gdc_patients')}")
        elif "gdc_overlapping" in intermediate:
            print(f"  GDC overlap with TCIA IDs: {intermediate.get('gdc_overlapping')}")
        if "fetch_requested" in intermediate:
            print(
                f"  Extract : requested={intermediate.get('fetch_requested')}, "
                f"radiomics_ok={intermediate.get('radiomics_ok')}, "
                f"failed={intermediate.get('radiomics_failed')}, "
                f"skipped={intermediate.get('imaging_skipped')}"
            )
    seg = payload.get("segmentation")
    if seg:
        print(f"Segmentation backend: {seg.get('backend')} (cuda={seg.get('cuda')})")

    if payload.get("patient_table"):
        print_patient_table(payload["patient_table"])

    specialists = payload.get("specialists") or {}
    if specialists and not payload.get("patient_table"):
        print(f"Specialists: {specialists.get('status')}")
        radio = specialists.get("radiomics") or {}
        geno = specialists.get("genomics") or {}
        if radio.get("feature_names"):
            print(f"  Radiomics features: {', '.join(radio['feature_names'][:8])}")
        elif specialists.get("radiomics_features"):
            print(f"  Radiomics features: {', '.join(specialists['radiomics_features'][:8])}")
        if geno.get("feature_names"):
            print(f"  Genomics features : {', '.join(geno['feature_names'][:8])}")
        elif specialists.get("genomics_features"):
            print(f"  Genomics features : {', '.join(specialists['genomics_features'][:8])}")

    stats = payload.get("stats") or {}
    associations = list(stats.get("associations") or payload.get("associations") or [])
    if not associations:
        associations = list(payload.get("top_associations") or [])
    associations = sorted(
        associations,
        key=lambda a: (abs(float(a.get("effect_size", 0.0))), float(a.get("q_value", 1.0))),
    )
    metrics = stats.get("metrics") or payload.get("metrics") or []
    prevalence = payload.get("mutation_prevalence") or {}

    def _fmt_r(r: float) -> str:
        if r < 0:
            return f"r={r:.3f} (|r|={abs(r):.3f})"
        return f"r={r:.3f}"

    def _gene_of(item: dict[str, Any]) -> str:
        g = str(item.get("genomic_feature") or "")
        if g.endswith("_mut") or g.endswith("_expr"):
            return g.rsplit("_", 1)[0]
        return g

    def _fmt_assoc(item: dict[str, Any]) -> str:
        r = float(item.get("effect_size", 0.0))
        gene = _gene_of(item)
        counts = prevalence.get(gene)
        mut = ""
        if counts:
            mut = (
                f"  [mut: altered={counts.get('altered', 0)}, "
                f"wildtype={counts.get('wildtype', 0)}]"
            )
        p = item.get("p_value")
        p_bit = f", p={float(p):.2e}" if p is not None else ""
        return (
            f"{item['imaging_feature']} ~ {item['genomic_feature']}: "
            f"{_fmt_r(r)}{p_bit}, q={float(item.get('q_value', 1.0)):.2e}, "
            f"n={item.get('n')}{mut}"
        )

    if associations or metrics:
        print("-" * 60)
        print(
            f"Associations ({len(associations)} total). "
            "CLI shows strongest 8 by |r| (ascending); full list in JSON."
        )
        print(
            "  Stats: r=Pearson correlation of radiomic feature vs mutation (0/1); "
            "|r| shown only if r<0; "
            "p=raw p-value; q=Benjamini–Hochberg FDR-adjusted p; "
            "n=patients in that test"
        )
        if not associations:
            print("  (none)")
        else:
            for item in associations[-8:]:
                print(f"  {_fmt_assoc(item)}")
        printable_metrics = [
            m for m in metrics if m.get("auroc") is not None or m.get("c_index") is not None
        ]
        if printable_metrics:
            print(
                "Metrics (genes with computable AUROC/C-index only)\n"
                "  AUROC=mutation prediction from radiomics; "
                "C-index=survival concordance if OS present; "
                "n_train/n_test=split sizes"
            )
            for item in printable_metrics:
                bits = [f"target={item.get('target')}"]
                if item.get("auroc") is not None:
                    bits.append(f"AUROC={item['auroc']:.3f}")
                if item.get("c_index") is not None:
                    bits.append(f"C-index={item['c_index']:.3f}")
                bits.append(f"n_train={item.get('n_train')}")
                bits.append(f"n_test={item.get('n_test')}")
                print(f"  {', '.join(bits)}")
        diagnostics = stats.get("diagnostics")
        if diagnostics and diagnostics.get("caveats"):
            print("Caveats")
            for caveat in diagnostics["caveats"]:
                print(f"  - {caveat}")

    literature = payload.get("literature") or {}
    if literature:
        print("-" * 60)
        print("Literature (annotation only; does not change associations)")
        src = literature.get("literature_source")
        n_prior = len(literature.get("prior_pairs") or [])
        if src or n_prior:
            print(f"  Source : {src or '?'} ({n_prior} prior radiomic–gene pairs)")
        if literature.get("search_query"):
            q = str(literature["search_query"])
            print(f"  PubMed : {q[:140]}{'...' if len(q) > 140 else ''}")
        print(
            "  Top 8 by |r| within each category (ascending |r|):\n"
            "    [supported]    disease-matched corpus paper agrees\n"
            "    [unverified]   no disease-matched paper in corpus\n"
            "    [contradicted] corpus paper argues against this pair\n"
            "    [note]         process reminder (not a finding class)"
        )

        def _with_mut_counts(text: str) -> str:
            if "_mut" not in text or not prevalence:
                return text
            try:
                right = text.split("~", 1)[1].strip()
                gene = right.split("_mut", 1)[0].strip().split()[0]
            except (IndexError, ValueError):
                return text
            counts = prevalence.get(gene)
            if not counts:
                return text
            return (
                f"{text}  [mut: altered={counts.get('altered', 0)}, "
                f"wildtype={counts.get('wildtype', 0)}]"
            )

        supported_items = [s for s in (literature.get("supports") or []) if s.get("supported")]
        if supported_items:
            print("  [supported]")
            for item in supported_items:
                print(f"    {_with_mut_counts(item.get('finding') or '')}")
                for paper in item.get("papers") or []:
                    print(f"             - {paper}")
        unverified_items = list(literature.get("unverified") or [])
        if unverified_items:
            print("  [unverified]")
            for item in unverified_items:
                print(f"    {_with_mut_counts(str(item))}")
        contra_items = list(literature.get("contradictions") or [])
        if contra_items:
            print("  [contradicted]")
            for item in contra_items:
                print(f"    {_with_mut_counts(str(item))}")
        for item in literature.get("proposed_refinements") or []:
            print(f"  [note] {item}")

    directive = payload.get("directive") or {}
    if directive or payload.get("looped"):
        print("-" * 60)
        print("Loop decision")
        print(f"  Iterations : {payload.get('iterations', 0)}")
        print(f"  Stopped    : {payload.get('stopped')}")
        print(f"  Action     : {directive.get('action')}")
        print(f"  Reason     : {directive.get('reason')}")
        print(f"  Human review required: {payload.get('human_review_required', directive.get('human_review_required'))}")
        print(f"  Auto-promoted        : {payload.get('auto_promoted', directive.get('auto_promoted'))}")
    note = format_dataset_extraction_note(payload.get("intermediate") or {})
    if note:
        print(note)
    print("=" * 60)


def build_catalog(name: str, *, download_dicom: bool = True) -> DemoCatalog | LiveCatalog:
    if name == "demo":
        return DemoCatalog()
    if name == "live":
        return LiveCatalog(download_dicom=download_dicom, extract_radiomics=download_dicom)
    raise ValueError(f"Unknown catalog '{name}'. Use demo or live.")


def _orchestrator(
    *,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> OrchestratorAgent:
    return OrchestratorAgent(
        tcga_project=tcga_project,
        tcia_collection=tcia_collection,
        modality=modality,
        genes=genes,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def _run_matcher(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    stage: int,
    download_dicom: bool = True,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    max_genes: int | None = None,
    min_altered: int = 1,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> tuple[dict[str, Any], ImageBundle | None, OmicsBundle | None]:
    catalog = build_catalog(catalog_name, download_dicom=download_dicom)
    gate = AlwaysAllowGate() if catalog_name == "demo" else FlagGate(approve_download)
    orchestrator = _orchestrator(
        tcga_project=tcga_project,
        tcia_collection=tcia_collection,
        modality=modality,
        genes=genes,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    request = orchestrator.parse_and_plan(question_text, disease=disease)
    request = request.model_copy(
        update={
            "max_patients": max_patients,
            "max_genes": max_genes,
            "min_altered": min_altered,
        }
    )
    matcher = DataMatcherAgent(
        catalog,
        gate=gate,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    preview = matcher.preview(request)
    payload: dict[str, Any] = {
        "stage": stage,
        "catalog": catalog_name,
        "question": orchestrator.parse(question_text, disease=disease).model_dump(),
        "request": request.model_dump(),
        "preview": preview.model_dump(),
        "source_counts": getattr(catalog, "last_source_counts", {}),
        "intermediate": getattr(catalog, "last_intermediate", {}),
        "paired_only": True,
        "token_required": False,
        "download_approved": bool(approve_download or catalog_name == "demo"),
    }
    try:
        images, omics = matcher.fetch(request)
    except DownloadDeniedError:
        payload["fetch"] = {
            "status": "blocked",
            "reason": "Human gate denied fetch. Re-run with --approve-download "
            "to pull metadata for the previewed IDs only (no DICOM, no BAM).",
        }
        return payload, None, None
    payload["fetch"] = {
        "status": "ok",
        "n_images": len(images.patient_ids),
        "n_omics": len(omics.patient_ids),
        "patient_ids": images.patient_ids,
        "series": [item.model_dump() for item in images.series],
        "mutations": omics.mutations,
        "clinical": omics.clinical,
        "imaging_notes": getattr(catalog, "last_imaging_notes", []),
    }
    return payload, images, omics


def run_stage1(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    download_dicom: bool = True,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    max_genes: int | None = None,
    min_altered: int = 1,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    payload, _, _ = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=1,
        download_dicom=download_dicom,
        tcga_project=tcga_project,
        tcia_collection=tcia_collection,
        modality=modality,
        genes=genes,
        max_genes=max_genes,
        min_altered=min_altered,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    return payload


def run_stage2(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    download_dicom: bool = True,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    max_genes: int | None = None,
    min_altered: int = 1,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    payload, images, omics = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=2,
        download_dicom=download_dicom,
        tcga_project=tcga_project,
        tcia_collection=tcia_collection,
        modality=modality,
        genes=genes,
        max_genes=max_genes,
        min_altered=min_altered,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    payload["joined"] = False
    if images is None or omics is None:
        payload["specialists"] = {"status": "blocked", "reason": "Stage 2 needs a successful Stage 1 fetch"}
        return payload
    outputs = extract_parallel(images, omics)
    payload["patient_table"] = build_patient_table(outputs.radiomics, outputs.genomics)
    payload["specialists"] = {
        "status": "ok" if outputs.radiomics and outputs.genomics else "partial",
        "imaging_error": outputs.imaging_error,
        "genomics_error": outputs.genomics_error,
        "radiomics": None
        if outputs.radiomics is None
        else {
            "n_patients": len(outputs.radiomics.patient_ids),
            "feature_names": outputs.radiomics.feature_names,
            "features": outputs.radiomics.features,
        },
        "genomics": None
        if outputs.genomics is None
        else {
            "n_patients": len(outputs.genomics.patient_ids),
            "feature_names": outputs.genomics.feature_names,
            "features": outputs.genomics.features,
        },
    }
    return payload


def run_stage3(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    download_dicom: bool = True,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    max_genes: int | None = None,
    min_altered: int = 1,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    payload, images, omics = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=3,
        download_dicom=download_dicom,
        tcga_project=tcga_project,
        tcia_collection=tcia_collection,
        modality=modality,
        genes=genes,
        max_genes=max_genes,
        min_altered=min_altered,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    payload["joined"] = True
    payload["looped"] = False
    if images is None or omics is None:
        payload["specialists"] = {"status": "blocked", "reason": "Stage 3 needs a successful Stage 1 fetch"}
        return payload
    outputs = extract_parallel(images, omics)
    payload["patient_table"] = build_patient_table(outputs.radiomics, outputs.genomics)
    payload["specialists"] = {
        "status": "ok" if outputs.radiomics and outputs.genomics else "partial",
        "imaging_error": outputs.imaging_error,
        "genomics_error": outputs.genomics_error,
        "radiomics_features": None if outputs.radiomics is None else outputs.radiomics.feature_names,
        "genomics_features": None if outputs.genomics is None else outputs.genomics.feature_names,
        "n_patients": len(outputs.patient_ids),
    }
    literature = LiteratureAgent(
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    literature.prepare(
        disease=str(payload["question"]["disease"]),
        question=question_text,
    )
    try:
        result = join_and_interpret(
            outputs,
            disease=payload["question"]["disease"],
            question=question_text,
            literature=literature,
        )
    except Stage3Error as exc:
        payload["stats"] = {"status": "blocked", "reason": str(exc)}
        return payload
    payload["stats"] = {
        "status": "ok",
        "associations": [item.model_dump() for item in result.stats.associations],
        "metrics": [item.model_dump() for item in result.stats.metrics],
        "diagnostics": result.stats.diagnostics.model_dump(),
        "promoted_findings": result.stats.promoted_findings,
    }
    payload["literature"] = result.literature.model_dump()
    return payload


def run_stage4(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    max_iterations: int,
    download_dicom: bool = True,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: str | None = None,
    max_genes: int | None = None,
    min_altered: int = 1,
    use_llm: bool | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    if catalog_name == "live" and not approve_download:
        raise ValueError("Live Stage 4 requires --approve-download (question-scoped DICOM + metadata).")
    catalog = build_catalog(catalog_name, download_dicom=download_dicom)
    gate = AlwaysAllowGate() if catalog_name == "demo" else FlagGate(approve_download)
    state = DiscoveryLoop(
        orchestrator=_orchestrator(
            tcga_project=tcga_project,
            tcia_collection=tcia_collection,
            modality=modality,
            genes=genes,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ),
        matcher=DataMatcherAgent(
            catalog,
            gate=gate,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ),
        imaging=ImagingRadiomicsAgent(),
        genomics=GenomicsAgent(),
        stats=StatisticalCriticalAgent(),
        literature=LiteratureAgent(
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
        ),
        max_iterations=max_iterations,
        max_patients=max_patients,
        max_genes=max_genes,
        min_altered=min_altered,
    ).run(question_text, disease=disease)
    patient_table = resolve_patient_table(
        state.radiomics,
        state.genomics,
        state.last_images,
        state.last_omics,
    )
    all_assoc = (
        []
        if state.stats is None
        else [item.model_dump() for item in state.stats.associations]
    )
    return {
        "stage": 4,
        "catalog": catalog_name,
        "joined": True,
        "looped": True,
        "question": state.question.model_dump(),
        "request": state.request.model_dump(),
        "request_history": [item.model_dump() for item in state.request_history],
        "iterations": state.iterations,
        "stopped": state.stopped,
        "directive": None if state.directive is None else state.directive.model_dump(),
        "directives": [item.model_dump() for item in state.directives],
        "human_review_required": True
        if state.directive is None
        else state.directive.human_review_required,
        "auto_promoted": [] if state.directive is None else state.directive.auto_promoted,
        "patient_table": patient_table,
        "mutation_prevalence": mutation_prevalence(patient_table),
        "specialists": {
            "imaging_error": state.imaging_error,
            "genomics_error": state.genomics_error,
            "n_radiomics_patients": len(state.radiomics.patient_ids)
            if state.radiomics
            else 0,
            "n_genomics_patients": len(state.genomics.patient_ids) if state.genomics else 0,
        },
        "top_associations": all_assoc[-5:] if all_assoc else [],
        "associations_by_gene": associations_by_gene(all_assoc, top_per_gene=0),
        "associations": all_assoc,
        "metrics": [] if state.stats is None else [item.model_dump() for item in state.stats.metrics],
        "literature": None if state.literature is None else state.literature.model_dump(),
        "literature_history": [item.model_dump() for item in state.history],
        "imaging_notes": getattr(catalog, "last_imaging_notes", []),
        "intermediate": getattr(catalog, "last_intermediate", {}),
        "segmentation": segmentation_backend(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic radiogenomics. Stage 4 joins stats with optional literature annotation (data-driven; literature does not alter the cohort)."
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="1 = matcher; 2 = specialists; 3 = stats+literature; 4 = full run (literature annotates only).",
    )
    parser.add_argument(
        "--question",
        default="Which imaging features are associated with EGFR mutations and survival in lung cancer?",
    )
    parser.add_argument(
        "--disease",
        default=None,
        help="Disease alias or site code (lung, breast, gbm, pancreas, kidney, ...). "
        "Also accepts TCGA project ids (TCGA-GBM). Use --list-diseases to list.",
    )
    parser.add_argument(
        "--tcga-project",
        default=None,
        help="Override GDC project (e.g. TCGA-GBM). Implies live pairing for that project.",
    )
    parser.add_argument(
        "--tcia-collection",
        default=None,
        help="Override TCIA collection (defaults to the TCGA project id).",
    )
    parser.add_argument("--modality", default=None, help="Override imaging modality (CT, MR, ...).")
    parser.add_argument(
        "--genes",
        choices=("auto",),
        default=None,
        help="Gene panel mode. Default: all genes mutated in the paired genomics cohort. "
        "--genes auto: literature/profile candidate genes only.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=None,
        help="Force Orchestrator + DataMatcher to use a free-tier LLM "
        "(Gemini 3.8 Flash by default). Requires GEMINI_API_KEY or GROQ_API_KEY.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force rule-based Orchestrator (no LLM calls).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("gemini", "groq"),
        default=None,
        help="LLM provider for Orchestrator. Default: gemini if GEMINI_API_KEY is set, else groq.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override model id (default: gemini-3.8-flash or openai/gpt-oss-120b).",
    )
    parser.add_argument(
        "--list-diseases",
        action="store_true",
        help="Print built-in TCGA disease aliases / projects and exit.",
    )
    parser.add_argument("--catalog", choices=("demo", "live"), default="demo")
    parser.add_argument(
        "--approve-download",
        action="store_true",
        help="Live catalog: allow question-scoped metadata/DICOM fetch for previewed IDs.",
    )
    parser.add_argument(
        "--no-dicom",
        action="store_true",
        help="Live catalog: metadata/mutations only (skip DICOM download and radiomics).",
    )
    parser.add_argument("--max-patients", type=int, default=24)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--out",
        default=None,
        help="Write full JSON to this path and a sibling .csv of all associations "
        "(e.g. outputs/stage4_lung.json -> outputs/stage4_lung.csv).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the full JSON payload to stdout (default: summary only).",
    )
    args = parser.parse_args()
    if args.list_diseases:
        from agentic_radiogen.schemas.profiles import get_profile, list_profiles, list_projects

        print("No fixed disease register. The agent matches your question to TCIA/GDC by keywords.")
        print("Example disease phrases:")
        for name in list_profiles():
            profile = get_profile(name)
            print(
                f"  {name:28s} modality={profile.default_modality:3s} "
                f"(sources discovered at runtime)"
            )
        print(f"Example projects often matched: {', '.join(list_projects())}")
        print("Override with --tcga-project / --tcia-collection if you need a fixed pair.")
        return
    download_dicom = not args.no_dicom
    if args.no_llm:
        use_llm: bool | None = False
    elif args.llm:
        use_llm = True
    else:
        use_llm = None  # auto: LLM when GEMINI_API_KEY / GROQ_API_KEY is set
    common = dict(
        disease=args.disease,
        catalog_name=args.catalog,
        approve_download=args.approve_download,
        max_patients=args.max_patients,
        download_dicom=download_dicom,
        tcga_project=args.tcga_project,
        tcia_collection=args.tcia_collection,
        modality=args.modality,
        genes=args.genes,
        use_llm=use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    try:
        if args.stage == 1:
            payload = run_stage1(args.question, **common)
        elif args.stage == 2:
            payload = run_stage2(args.question, **common)
        elif args.stage == 3:
            payload = run_stage3(args.question, **common)
        else:
            payload = run_stage4(
                args.question,
                max_iterations=args.max_iterations,
                **common,
            )
    except (RemoteApiError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc

    print_summary(payload)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Full JSON written to: {out_path.resolve()}")
        associations = list(
            (payload.get("stats") or {}).get("associations")
            or payload.get("associations")
            or []
        )
        if associations:
            csv_path = out_path.with_suffix(".csv")
            disease = str((payload.get("question") or {}).get("disease") or "unknown")
            n_rows = write_associations_csv(
                csv_path,
                associations,
                disease=disease,
                mutation_prevalence=payload.get("mutation_prevalence") or {},
                literature=payload.get("literature"),
            )
            print(f"Associations CSV ({n_rows} rows) written to: {csv_path.resolve()}")
    if args.json:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()