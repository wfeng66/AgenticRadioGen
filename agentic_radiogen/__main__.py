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
)
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
    print(f"Genes   : {', '.join(question.get('genes') or []) or '(profile default)'}")
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
    associations = stats.get("associations") or payload.get("top_associations") or []
    by_gene = payload.get("associations_by_gene") or associations_by_gene(associations)
    metrics = stats.get("metrics") or payload.get("metrics") or []
    if associations or by_gene or metrics:
        print("-" * 60)
        print("Imaging feature ~ genomic alteration (top per gene)")
        if not by_gene:
            print("  (none)")
        for gene in sorted(by_gene):
            print(f"  [{gene}]")
            for item in by_gene[gene]:
                print(
                    f"    {item['imaging_feature']}: "
                    f"r={item['effect_size']:.3f}, q={item['q_value']:.2e}, n={item['n']}"
                )
        print("Metrics (per genomic target)")
        if not metrics:
            print("  (none)")
        for item in metrics:
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
        print("Literature")
        for item in literature.get("supports") or []:
            mark = "supported" if item.get("supported") else "not supported"
            print(f"  [{mark}] {item.get('finding')}")
            for paper in item.get("papers") or []:
                print(f"           - {paper}")
        for item in literature.get("unverified") or []:
            print(f"  [unverified] {item}")
        for item in literature.get("contradictions") or []:
            print(f"  [contradiction] {item}")
        for item in literature.get("proposed_refinements") or []:
            print(f"  [refinement] {item}")

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
    print("=" * 60)


def build_catalog(name: str, *, download_dicom: bool = True) -> DemoCatalog | LiveCatalog:
    if name == "demo":
        return DemoCatalog()
    if name == "live":
        return LiveCatalog(download_dicom=download_dicom, extract_radiomics=download_dicom)
    raise ValueError(f"Unknown catalog '{name}'. Use demo or live.")


def _run_matcher(
    question_text: str,
    *,
    disease: str | None,
    catalog_name: str,
    approve_download: bool,
    max_patients: int,
    stage: int,
    download_dicom: bool = True,
) -> tuple[dict[str, Any], ImageBundle | None, OmicsBundle | None]:
    catalog = build_catalog(catalog_name, download_dicom=download_dicom)
    gate = AlwaysAllowGate() if catalog_name == "demo" else FlagGate(approve_download)
    orchestrator = OrchestratorAgent()
    request = orchestrator.parse_and_plan(question_text, disease=disease)
    request = request.model_copy(update={"max_patients": max_patients})
    matcher = DataMatcherAgent(catalog, gate=gate)
    preview = matcher.preview(request)
    payload: dict[str, Any] = {
        "stage": stage,
        "catalog": catalog_name,
        "question": orchestrator.parse(question_text, disease=disease).model_dump(),
        "request": request.model_dump(),
        "preview": preview.model_dump(),
        "source_counts": getattr(catalog, "last_source_counts", {}),
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
) -> dict[str, Any]:
    payload, _, _ = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=1,
        download_dicom=download_dicom,
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
) -> dict[str, Any]:
    payload, images, omics = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=2,
        download_dicom=download_dicom,
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
) -> dict[str, Any]:
    payload, images, omics = _run_matcher(
        question_text,
        disease=disease,
        catalog_name=catalog_name,
        approve_download=approve_download,
        max_patients=max_patients,
        stage=3,
        download_dicom=download_dicom,
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
    try:
        result = join_and_interpret(outputs, disease=payload["question"]["disease"])
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
) -> dict[str, Any]:
    if catalog_name == "live" and not approve_download:
        raise ValueError("Live Stage 4 requires --approve-download (question-scoped DICOM + metadata).")
    catalog = build_catalog(catalog_name, download_dicom=download_dicom)
    gate = AlwaysAllowGate() if catalog_name == "demo" else FlagGate(approve_download)
    state = DiscoveryLoop(
        orchestrator=OrchestratorAgent(),
        matcher=DataMatcherAgent(catalog, gate=gate),
        imaging=ImagingRadiomicsAgent(),
        genomics=GenomicsAgent(),
        stats=StatisticalCriticalAgent(),
        literature=LiteratureAgent(),
        max_iterations=max_iterations,
        max_patients=max_patients,
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
        "top_associations": all_assoc[:5],
        "associations_by_gene": associations_by_gene(all_assoc, top_per_gene=3),
        "associations": all_assoc,
        "metrics": [] if state.stats is None else [item.model_dump() for item in state.stats.metrics],
        "literature": None if state.literature is None else state.literature.model_dump(),
        "literature_history": [item.model_dump() for item in state.history],
        "imaging_notes": getattr(catalog, "last_imaging_notes", []),
        "segmentation": segmentation_backend(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic radiogenomics. Stage 4 runs the literature-driven self-correction loop."
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="1 = matcher; 2 = specialists; 3 = stats+literature; 4 = self-correction loop.",
    )
    parser.add_argument(
        "--question",
        default="Which imaging features are associated with EGFR mutations and survival in lung cancer?",
    )
    parser.add_argument("--disease", default=None, help="Optional profile: lung or breast")
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
        help="Write the full JSON payload to this file (e.g. outputs/stage4_lung.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the full JSON payload to stdout (default: summary only).",
    )
    args = parser.parse_args()
    download_dicom = not args.no_dicom

    try:
        if args.stage == 1:
            payload = run_stage1(
                args.question,
                disease=args.disease,
                catalog_name=args.catalog,
                approve_download=args.approve_download,
                max_patients=args.max_patients,
                download_dicom=download_dicom,
            )
        elif args.stage == 2:
            payload = run_stage2(
                args.question,
                disease=args.disease,
                catalog_name=args.catalog,
                approve_download=args.approve_download,
                max_patients=args.max_patients,
                download_dicom=download_dicom,
            )
        elif args.stage == 3:
            payload = run_stage3(
                args.question,
                disease=args.disease,
                catalog_name=args.catalog,
                approve_download=args.approve_download,
                max_patients=args.max_patients,
                download_dicom=download_dicom,
            )
        else:
            payload = run_stage4(
                args.question,
                disease=args.disease,
                catalog_name=args.catalog,
                approve_download=args.approve_download,
                max_patients=args.max_patients,
                max_iterations=args.max_iterations,
                download_dicom=download_dicom,
            )
    except (RemoteApiError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print_summary(payload)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Full JSON written to: {out_path.resolve()}")
    if args.json:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()