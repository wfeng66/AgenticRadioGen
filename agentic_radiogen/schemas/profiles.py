from __future__ import annotations

from pydantic import BaseModel, Field

# Pan-cancer defaults when a disease has no curated panel.
PANCAN_GENES = [
    "TP53",
    "KRAS",
    "PIK3CA",
    "PTEN",
    "EGFR",
    "BRAF",
    "IDH1",
    "CDKN2A",
    "MYC",
    "RB1",
]

_PANCAN_ALIASES = {
    "tp53": "TP53",
    "p53": "TP53",
    "kras": "KRAS",
    "pik3ca": "PIK3CA",
    "pten": "PTEN",
    "egfr": "EGFR",
    "braf": "BRAF",
    "idh1": "IDH1",
    "cdkn2a": "CDKN2A",
    "myc": "MYC",
    "rb1": "RB1",
}


class DiseaseProfile(BaseModel):
    """Reusable disease configuration. Agents stay generic; only this changes."""

    name: str
    tcga_project: str
    tcia_collection: str
    default_modality: str
    candidate_genes: list[str]
    gene_aliases: dict[str, str] = Field(default_factory=dict)
    immune_signatures: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    imaging_feature_hints: list[str] = Field(default_factory=list)


def _entry(
    project: str,
    *,
    aliases: tuple[str, ...],
    modality: str = "CT",
    genes: list[str] | None = None,
    gene_aliases: dict[str, str] | None = None,
    tcia_collection: str | None = None,
    gdc_project: str | None = None,
) -> DiseaseProfile:
    """One disease. Collection defaults to the project id; overrides allow CPTAC-style pairing."""
    slug = project.replace("TCGA-", "").lower()
    name = aliases[0] if aliases else slug
    genes = list(genes or PANCAN_GENES)
    aliases_map = dict(_PANCAN_ALIASES)
    aliases_map.update({g.lower(): g for g in genes})
    if gene_aliases:
        aliases_map.update(gene_aliases)
    gdc = (gdc_project or project).strip()
    tcia = (tcia_collection or project).strip()
    return DiseaseProfile(
        name=name,
        tcga_project=gdc,
        tcia_collection=tcia,
        default_modality=modality,
        candidate_genes=genes,
        gene_aliases=aliases_map,
        immune_signatures=["TMB", "CD8_Tcell"],
        endpoints=["OS", "PFS", "subtype"],
        imaging_feature_hints=["shape", "texture", "intensity"],
    )


# All major TCGA projects that commonly have TCIA imaging under the same collection id.
# Friendly aliases (first = canonical disease name) remove the need to register per disease.
_TCGA_SPECS: list[dict] = [
    {
        "project": "TCGA-ACC",
        "aliases": ("adrenal", "acc", "adrenocortical"),
        "genes": ["TP53", "CTNNB1", "MEN1", "PRKAR1A", "ZNRF3"],
    },
    {
        "project": "TCGA-BLCA",
        "aliases": ("bladder", "blca", "urothelial"),
        "genes": ["TP53", "FGFR3", "PIK3CA", "RB1", "KDM6A", "ARID1A"],
    },
    {
        "project": "TCGA-BRCA",
        "aliases": ("breast", "brca", "mammary"),
        "modality": "MR",
        "genes": ["BRCA1", "BRCA2", "ESR1", "ERBB2", "PGR", "TP53", "PIK3CA"],
        "gene_aliases": {
            "her2": "ERBB2",
            "erbb2": "ERBB2",
            "er": "ESR1",
            "esr1": "ESR1",
            "pr": "PGR",
            "pgr": "PGR",
            "brca1": "BRCA1",
            "brca2": "BRCA2",
        },
    },
    {
        "project": "TCGA-CESC",
        "aliases": ("cervical", "cesc", "cervix"),
        "genes": ["PIK3CA", "TP53", "KRAS", "PTEN", "EGFR"],
    },
    {
        "project": "TCGA-CHOL",
        "aliases": ("cholangiocarcinoma", "chol", "bile_duct"),
        "genes": ["TP53", "KRAS", "IDH1", "BAP1", "ARID1A"],
    },
    {
        "project": "TCGA-COAD",
        "aliases": ("colon", "coad", "colorectal", "crc"),
        "genes": ["APC", "TP53", "KRAS", "PIK3CA", "BRAF", "SMAD4"],
        "gene_aliases": {"apc": "APC", "smad4": "SMAD4"},
    },
    {
        "project": "TCGA-DLBC",
        "aliases": ("lymphoma", "dlbc", "dlbcl"),
        "modality": "CT",
        "genes": ["TP53", "MYC", "BCL2", "CDKN2A", "EZH2"],
    },
    {
        "project": "TCGA-ESCA",
        "aliases": ("esophageal", "esca", "esophagus"),
        "genes": ["TP53", "CDKN2A", "PIK3CA", "NOTCH1", "NFE2L2"],
    },
    {
        "project": "TCGA-GBM",
        "aliases": ("gbm", "glioblastoma", "glioma"),
        "modality": "MR",
        "genes": ["IDH1", "TP53", "EGFR", "PTEN", "CDKN2A", "NF1"],
        "gene_aliases": {"idh1": "IDH1", "nf1": "NF1"},
    },
    {
        "project": "TCGA-HNSC",
        "aliases": ("head_neck", "hnsc", "hnscc", "head and neck"),
        "genes": ["TP53", "CDKN2A", "PIK3CA", "NOTCH1", "FAT1"],
    },
    {
        "project": "TCGA-KICH",
        "aliases": ("kidney_chromophobe", "kich"),
        "genes": ["TP53", "PTEN", "CDKN2A", "RB1"],
    },
    {
        "project": "TCGA-KIRC",
        "aliases": ("kidney", "kirc", "renal", "rcc", "clear_cell_kidney"),
        "genes": ["VHL", "PBRM1", "SETD2", "BAP1", "TP53", "MTOR"],
        "gene_aliases": {"vhl": "VHL", "pbrm1": "PBRM1", "setd2": "SETD2", "mtor": "MTOR"},
    },
    {
        "project": "TCGA-KIRP",
        "aliases": ("kidney_papillary", "kirp"),
        "genes": ["MET", "NF2", "SMARCB1", "TP53", "CDKN2A"],
        "gene_aliases": {"met": "MET", "nf2": "NF2"},
    },
    {
        "project": "TCGA-LGG",
        "aliases": ("lgg", "low_grade_glioma", "lower_grade_glioma"),
        "modality": "MR",
        "genes": ["IDH1", "TP53", "ATRX", "CIC", "FUBP1", "EGFR"],
        "gene_aliases": {"idh1": "IDH1", "atrx": "ATRX"},
    },
    {
        "project": "TCGA-LIHC",
        "aliases": ("liver", "lihc", "hcc", "hepatocellular"),
        "genes": ["TP53", "CTNNB1", "AXIN1", "ARID1A", "CDKN2A"],
        "gene_aliases": {"ctnnb1": "CTNNB1"},
    },
    {
        "project": "TCGA-LUAD",
        "aliases": ("lung", "luad", "adenocarcinoma of the lung", "lung adenocarcinoma"),
        "genes": ["EGFR", "KRAS", "TP53", "STK11", "KEAP1", "ALK"],
        "gene_aliases": {
            "egfr": "EGFR",
            "kras": "KRAS",
            "tp53": "TP53",
            "p53": "TP53",
            "stk11": "STK11",
            "alk": "ALK",
        },
    },
    {
        "project": "TCGA-LUSC",
        "aliases": ("lung_squamous", "lusc", "squamous lung"),
        "genes": ["TP53", "CDKN2A", "PIK3CA", "PTEN", "NFE2L2", "SOX2"],
    },
    {
        "project": "TCGA-MESO",
        "aliases": ("mesothelioma", "meso"),
        "genes": ["BAP1", "NF2", "TP53", "SETD2", "CDKN2A"],
    },
    {
        "project": "TCGA-OV",
        "aliases": ("ovarian", "ov", "ovary"),
        "modality": "CT",
        "genes": ["TP53", "BRCA1", "BRCA2", "NF1", "RB1", "CDK12"],
        "gene_aliases": {"brca1": "BRCA1", "brca2": "BRCA2"},
    },
    {
        "project": "TCGA-PAAD",
        "aliases": ("pancreas", "paad", "pancreatic", "pdac", "pancreatic ductal adenocarcinoma"),
        "genes": ["KRAS", "TP53", "CDKN2A", "SMAD4", "ARID1A"],
        "gene_aliases": {"smad4": "SMAD4"},
        # TCGA-PAAD has genomics on GDC but no public TCIA imaging under that name.
        # CPTAC-PDA (TCIA) ∩ CPTAC-3 (GDC) is the paired live path for PDAC.
        "tcia_collection": "CPTAC-PDA",
        "gdc_project": "CPTAC-3",
    },
    {
        "project": "TCGA-PCPG",
        "aliases": ("pheochromocytoma", "pcpg", "paraganglioma"),
        "genes": ["RET", "VHL", "NF1", "SDHB", "SDHD", "EPAS1"],
    },
    {
        "project": "TCGA-PRAD",
        "aliases": ("prostate", "prad"),
        "modality": "MR",
        "genes": ["TP53", "PTEN", "SPOP", "FOXA1", "AR", "BRCA2"],
        "gene_aliases": {"ar": "AR", "spop": "SPOP", "foxa1": "FOXA1"},
    },
    {
        "project": "TCGA-READ",
        "aliases": ("rectal", "read", "rectum"),
        "genes": ["APC", "TP53", "KRAS", "PIK3CA", "BRAF", "SMAD4"],
    },
    {
        "project": "TCGA-SARC",
        "aliases": ("sarcoma", "sarc"),
        "genes": ["TP53", "ATRX", "RB1", "CDKN2A", "NF1"],
    },
    {
        "project": "TCGA-SKCM",
        "aliases": ("melanoma", "skcm", "skin"),
        "genes": ["BRAF", "NRAS", "TP53", "CDKN2A", "PTEN", "NF1"],
        "gene_aliases": {"nras": "NRAS", "braf": "BRAF"},
    },
    {
        "project": "TCGA-STAD",
        "aliases": ("stomach", "stad", "gastric"),
        "genes": ["TP53", "ARID1A", "PIK3CA", "CDH1", "KRAS", "RHOA"],
    },
    {
        "project": "TCGA-TGCT",
        "aliases": ("testicular", "tgct", "germ_cell"),
        "genes": ["KIT", "KRAS", "NRAS", "TP53", "MDM2"],
    },
    {
        "project": "TCGA-THCA",
        "aliases": ("thyroid", "thca"),
        "genes": ["BRAF", "NRAS", "HRAS", "RET", "TP53", "PTEN"],
        "gene_aliases": {"hras": "HRAS", "ret": "RET"},
    },
    {
        "project": "TCGA-THYM",
        "aliases": ("thymoma", "thym"),
        "genes": ["GTF2I", "HRAS", "NRAS", "TP53"],
    },
    {
        "project": "TCGA-UCEC",
        "aliases": ("endometrial", "ucec", "uterine", "uterus"),
        "genes": ["PTEN", "PIK3CA", "TP53", "ARID1A", "CTNNB1", "KRAS"],
    },
    {
        "project": "TCGA-UCS",
        "aliases": ("uterine_carcinosarcoma", "ucs"),
        "genes": ["TP53", "PIK3CA", "FBXW7", "PTEN", "KRAS"],
    },
    {
        "project": "TCGA-UVM",
        "aliases": ("uveal_melanoma", "uvm"),
        "genes": ["GNAQ", "GNA11", "BAP1", "SF3B1", "EIF1AX"],
    },
]


def _build_registry() -> tuple[dict[str, DiseaseProfile], dict[str, str]]:
    by_name: dict[str, DiseaseProfile] = {}
    alias_to_name: dict[str, str] = {}
    for spec in _TCGA_SPECS:
        profile = _entry(
            spec["project"],
            aliases=tuple(spec["aliases"]),
            modality=spec.get("modality", "CT"),
            genes=spec.get("genes"),
            gene_aliases=spec.get("gene_aliases"),
            tcia_collection=spec.get("tcia_collection"),
            gdc_project=spec.get("gdc_project"),
        )
        by_name[profile.name] = profile
        # Also index by project code and bare site code (e.g. luad, TCGA-LUAD).
        slug = profile.tcga_project.replace("TCGA-", "").lower()
        alias_to_name[profile.name.lower()] = profile.name
        alias_to_name[slug] = profile.name
        alias_to_name[profile.tcga_project.lower()] = profile.name
        for alias in spec["aliases"]:
            alias_to_name[alias.lower()] = profile.name
    return by_name, alias_to_name


_REGISTRY, _ALIASES = _build_registry()

# Back-compat exports used by tests.
LUNG_PROFILE = _REGISTRY["lung"]
BREAST_PROFILE = _REGISTRY["breast"]


def _dynamic_from_project(project: str, *, modality: str = "CT") -> DiseaseProfile:
    project = project.strip().upper()
    if not project.startswith("TCGA-"):
        project = f"TCGA-{project}"
    slug = project.replace("TCGA-", "").lower()
    return _entry(project, aliases=(slug,), modality=modality, genes=list(PANCAN_GENES))


def get_profile(
    name: str,
    *,
    tcga_project: str | None = None,
    tcia_collection: str | None = None,
    modality: str | None = None,
    genes: list[str] | None = None,
) -> DiseaseProfile:
    """Resolve a disease by alias, site code, or TCGA project id.

    Curated TCGA sites need no extra registration. Any ``TCGA-*`` / site code also
    resolves dynamically (pan-cancer gene panel) so new collections work without code changes.
    """
    raw = name.strip()
    key = raw.lower()
    # NSCLC is not an alias of LUAD; sources are resolved by keyword match in LiveCatalog.
    if key in {
        "nsclc",
        "non-small cell",
        "non small cell",
        "non-small-cell",
        "non-small cell lung",
        "non small cell lung",
        "non-small cell lung cancer",
        "non small cell lung cancer",
        "non-small-cell lung cancer",
    }:
        base = _REGISTRY["lung"]
        profile = base.model_copy(
            update={
                "name": "nsclc",
                # Placeholder; LiveCatalog replaces via TCIA/GDC keyword discovery.
                "tcga_project": "KEYWORD",
                "tcia_collection": "KEYWORD",
            }
        )
    elif key in _ALIASES:
        profile = _REGISTRY[_ALIASES[key]]
    elif key.startswith("tcga-"):
        project = "TCGA-" + key.split("-", 1)[1].upper()
        matched = next((p for p in _REGISTRY.values() if p.tcga_project == project), None)
        profile = matched or _dynamic_from_project(project, modality=modality or "CT")
    else:
        project = f"TCGA-{raw.upper()}"
        matched = next((p for p in _REGISTRY.values() if p.tcga_project == project), None)
        if matched is not None:
            profile = matched
        elif raw.replace("_", "").replace("-", "").isalnum():
            profile = _dynamic_from_project(project, modality=modality or "CT")
        else:
            known = ", ".join(list_profiles()[:12]) + ", ..."
            raise KeyError(
                f"Unknown disease '{name}'. Use a disease name, site code (e.g. gbm, paad), "
                f"or TCGA project (e.g. TCGA-GBM). Examples: {known}"
            )

    updates: dict = {}
    if tcga_project:
        project = tcga_project.strip().upper()
        if not project.startswith("TCGA-"):
            project = f"TCGA-{project}"
        updates["tcga_project"] = project
        if not tcia_collection:
            updates["tcia_collection"] = project
    if tcia_collection:
        updates["tcia_collection"] = tcia_collection.strip()
    if modality:
        updates["default_modality"] = modality.strip().upper()
    if genes:
        updates["candidate_genes"] = list(genes)
        alias_map = dict(profile.gene_aliases)
        alias_map.update({g.lower(): g for g in genes})
        updates["gene_aliases"] = alias_map
    return profile.model_copy(update=updates) if updates else profile


def list_profiles() -> list[str]:
    return sorted(_REGISTRY)


def list_projects() -> list[str]:
    return sorted({p.tcga_project for p in _REGISTRY.values()})


def disease_hint_map() -> dict[str, tuple[str, ...]]:
    """Canonical disease name → text hints for question parsing."""
    out: dict[str, tuple[str, ...]] = {}
    for spec in _TCGA_SPECS:
        name = spec["aliases"][0]
        # Longer phrases first help matching; include project slug.
        hints = tuple(sorted(set(spec["aliases"]), key=len, reverse=True))
        out[name] = hints
    return out
