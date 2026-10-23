"""Binary gene labels follow the somatic-MAF qualifying Variant_Classification set."""

from __future__ import annotations

from agentic_radiogen.data.gdc_client import (
    QUALIFYING_CONSEQUENCE_TYPES,
    QUALIFYING_MAF_VARIANT_CLASSIFICATIONS,
    GdcClient,
    _qualifying_gene_symbols,
    _tcga_patient_id,
)


def test_qualifying_sets_match_prespecified_maf_definition() -> None:
    assert "Missense_Mutation" in QUALIFYING_MAF_VARIANT_CLASSIFICATIONS
    assert "Silent" not in QUALIFYING_MAF_VARIANT_CLASSIFICATIONS
    assert "missense_variant" in QUALIFYING_CONSEQUENCE_TYPES
    assert "synonymous_variant" not in QUALIFYING_CONSEQUENCE_TYPES


def test_tcga_barcode_maps_to_patient_id() -> None:
    assert _tcga_patient_id("TCGA-38-6178-01A-11R-1755-07") == "TCGA-38-6178"
    assert _tcga_patient_id("TCGA-38-6178") == "TCGA-38-6178"


def test_synonymous_only_does_not_label_mutant() -> None:
    hit = {
        "case": {"submitter_id": "TCGA-05-4244"},
        "ssm": {
            "consequence": [
                {
                    "transcript": {
                        "gene": {"symbol": "TP53"},
                        "consequence_type": "synonymous_variant",
                    }
                },
                {
                    "transcript": {
                        "gene": {"symbol": "TP53"},
                        "consequence_type": "intron_variant",
                    }
                },
            ]
        },
    }
    assert _qualifying_gene_symbols(hit) == set()


def test_missense_labels_mutant() -> None:
    hit = {
        "case": {"submitter_id": "TCGA-05-4244"},
        "ssm": {
            "consequence": [
                {
                    "transcript": {
                        "gene": {"symbol": "TP53"},
                        "consequence_type": "intron_variant",
                    }
                },
                {
                    "transcript": {
                        "gene": {"symbol": "TP53"},
                        "consequence_type": "missense_variant",
                    }
                },
            ]
        },
    }
    assert _qualifying_gene_symbols(hit) == {"TP53"}


def test_mutation_flags_require_qualifying_somatic_consequence() -> None:
    captured: list[dict] = []

    def poster(url: str, payload: dict):
        assert "ssm_occurrences" in url
        captured.append(payload)
        return {
            "data": {
                "hits": [
                    {
                        "case": {
                            "submitter_id": "TCGA-05-4244",
                            "observation": {
                                "mutation_status": "somatic",
                                "sample": {
                                    "tumor_sample_barcode": "TCGA-05-4244-01A-01D-1105-08"
                                },
                            },
                        },
                        "ssm": {
                            "consequence": [
                                {
                                    "transcript": {
                                        "gene": {"symbol": "TP53"},
                                        "consequence_type": "missense_variant",
                                    }
                                }
                            ]
                        },
                    },
                    {
                        "case": {"submitter_id": "TCGA-05-4249"},
                        "ssm": {
                            "consequence": [
                                {
                                    "transcript": {
                                        "gene": {"symbol": "TP53"},
                                        "consequence_type": "synonymous_variant",
                                    }
                                }
                            ]
                        },
                    },
                ],
                "pagination": {"total": 2},
            }
        }

    flags = GdcClient(poster=poster).mutation_flags(
        ["TCGA-05-4244", "TCGA-05-4249"], ["TP53"]
    )
    assert flags["TCGA-05-4244"]["TP53"] == 1
    assert flags["TCGA-05-4249"]["TP53"] == 0

    filt = captured[0]["filters"]
    assert filt["op"] == "and"
    fields = {c["content"]["field"] for c in filt["content"]}
    assert "case.observation.mutation_status" in fields
    assert "ssm.consequence.transcript.consequence_type" in fields
