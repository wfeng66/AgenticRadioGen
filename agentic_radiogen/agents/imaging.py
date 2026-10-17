from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_radiogen.schemas.contracts import ImageBundle, RadiomicMatrix

# Literature often says "texture" / "shape" / "intensity"; PyRadiomics uses family tokens.
_RADIOMIC_SYNONYMS: dict[str, frozenset[str]] = {
    "texture": frozenset(
        {
            "texture",
            "glcm",
            "gldm",
            "glrlm",
            "glszm",
            "ngtdm",
            "entropy",
            "contrast",
            "dissimilarity",
            "homogeneity",
            "correlation",
            "asm",
            "energy",
            "idm",
            "differenceaverage",
            "differenceentropy",
            "dependence",
            "zone",
            "run",
            "graylevel",
            "smalldependence",
            "largedependence",
        }
    ),
    "shape": frozenset(
        {
            "shape",
            "sphericity",
            "elongation",
            "flatness",
            "compactness",
            "surface",
            "volume",
            "mesh",
            "voxelvolume",
            "surfacearea",
            "maximum3ddiameter",
            "minoraxis",
            "majoraxis",
        }
    ),
    "firstorder": frozenset(
        {
            "firstorder",
            "intensity",
            "mean",
            "median",
            "skewness",
            "kurtosis",
            "percentile",
            "variance",
            "robustmeanabsolutedeviation",
            "rootmeansquared",
            "uniformity",
            "energy",
            "totalenergy",
            "minimum",
            "maximum",
            "range",
            "interquartilerange",
        }
    ),
    "wavelet": frozenset({"wavelet", "ll", "lh", "hl", "hh"}),
}


class ImagingRadiomicsAgent:
    """Consumes an ImageBundle only. Never calls the genomics agent.

    Uses precomputed radiomics from the matcher when present (live fetch).
    Re-reads DICOM only when precomputed features are missing.
    Also maps PyRadiomics feature names ↔ literature radiomic terms.
    """

    def extract(self, bundle: ImageBundle) -> RadiomicMatrix:
        if not bundle.series:
            raise ValueError("ImageBundle has no series")
        features: dict[str, dict[str, float]] = {}
        names: set[str] = set()
        skipped: list[str] = []
        for series in bundle.series:
            try:
                row = self._series_features(series)
            except Exception as exc:
                skipped.append(f"{series.patient_id}: {exc}")
                continue
            features[series.patient_id] = row
            names.update(row)
        if not features:
            detail = "; ".join(skipped[:5])
            raise ValueError(
                f"No radiomic payload for any series"
                + (f" ({detail})" if detail else "")
            )
        ordered = sorted(names)
        return RadiomicMatrix(
            patient_ids=sorted(features),
            feature_names=ordered,
            features=features,
        )

    def feature_tokens(self, imaging_feature: str) -> set[str]:
        """Tokenize a PyRadiomics-style feature name."""
        parts = imaging_feature.lower().replace("-", "_").split("_")
        return {p for p in parts if p}

    def expand_literature_terms(self, terms: set[str] | frozenset[str] | list[str]) -> set[str]:
        """Expand literature tokens (e.g. texture) into matchable PyRadiomics families."""
        expanded: set[str] = set()
        for raw in terms:
            term = str(raw).strip().lower().replace("-", "").replace(" ", "")
            if not term:
                continue
            expanded.add(term)
            # Also keep underscore form for feature-token overlap
            expanded.add(str(raw).strip().lower().replace("-", "_"))
            for canon, members in _RADIOMIC_SYNONYMS.items():
                if term == canon or term in members:
                    expanded |= set(members)
                    expanded.add(canon)
        return {t for t in expanded if t}

    def relate_to_literature_terms(
        self,
        imaging_feature: str,
        literature_terms: set[str] | frozenset[str] | list[str],
    ) -> tuple[bool, str, float]:
        """Return (related?, explanation, score in [0,1]) vs literature radiomic terms."""
        feat_tokens = self.feature_tokens(imaging_feature)
        if not feat_tokens:
            return False, "no feature tokens", 0.0
        lit = {str(t).strip().lower() for t in literature_terms if str(t).strip()}
        if not lit:
            return False, "no literature radiomic terms", 0.0

        expanded = self.expand_literature_terms(lit)
        # Drop ultra-generic tokens that alone never prove relatedness
        generic = {
            "original",
            "radiomic",
            "radiomics",
            "feature",
            "features",
            "imaging",
            "ct",
            "mri",
            "mr",
            "cancer",
            "tumor",
            "tumour",
        }
        specific_lit = expanded - generic
        if not specific_lit:
            return False, "literature terms too generic", 0.0

        hits: set[str] = set()
        for term in specific_lit:
            if term in feat_tokens:
                hits.add(term)
                continue
            if len(term) >= 3 and any(term in tok or tok in term for tok in feat_tokens):
                hits.add(term)

        if not hits:
            return False, "no radiomic family overlap", 0.0

        # Prefer hits that came from expanding a literary synonym (texture→gldm)
        lit_raw = {t.replace("-", "").replace(" ", "") for t in lit}
        via_synonym = bool(hits - lit_raw) and not (hits & lit_raw)
        score = min(1.0, 0.55 + 0.15 * len(hits))
        if via_synonym:
            reason = (
                f"feature tokens {sorted(feat_tokens)[:6]} relate to literature "
                f"{sorted(lit)[:6]} via families {sorted(hits)[:6]}"
            )
        else:
            reason = (
                f"feature tokens {sorted(feat_tokens)[:6]} overlap literature "
                f"terms/families {sorted(hits)[:6]}"
            )
        return True, reason, score

    def _series_features(self, series) -> dict[str, float]:
        row = {
            k: float(v)
            for k, v in (series.precomputed_features or {}).items()
            if not str(k).startswith("meta_")
        }
        if not row and series.local_path:
            disease = getattr(series, "disease", None)
            modality = getattr(series, "modality", None) or "CT"
            row.update(
                self._from_local_path(
                    series.local_path, disease=disease, modality=modality
                )
            )
        if series.volume_summary:
            row.setdefault(
                "original_firstorder_Mean", float(series.volume_summary.get("mean", 0.0))
            )
            row.setdefault(
                "original_firstorder_Std", float(series.volume_summary.get("std", 0.0))
            )
            row.setdefault(
                "original_shape_VoxelVolume", float(series.volume_summary.get("size", 0.0))
            )
        if not row:
            raise ValueError(f"No radiomic payload for series {series.series_uid}")
        return {k: float(v) for k, v in row.items()}

    @staticmethod
    def _from_local_path(
        local_path: str,
        *,
        disease: str | None = None,
        modality: str = "CT",
    ) -> dict[str, float]:
        path = Path(local_path)
        if path.is_dir():
            from agentic_radiogen.imaging.radiomics_extract import extract_series_features

            extracted = extract_series_features(
                path, disease=disease, modality=modality
            )
            return {
                k: float(v)
                for k, v in extracted["features"].items()
                if not str(k).startswith("meta_")
            }
        if path.suffix == ".npy":
            volume = np.load(path)
            from agentic_radiogen.imaging.radiomics_extract import extract_radiomics_from_volume

            features, _ = extract_radiomics_from_volume(
                volume, disease=disease, modality=modality
            )
            return {
                k: float(v)
                for k, v in features.items()
                if not str(k).startswith("meta_")
            }
        raise ValueError(f"Unsupported imaging path: {local_path}")
