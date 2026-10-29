# AgenticRadioGen

Disease-agnostic **agentic radiogenomics** pipeline: given a research question (e.g. breast or lung cancer), the system matches **TCIA imaging** with **GDC genomics**, extracts radiomic and mutation features, tests imaging–gene associations, and optionally annotates findings with a small literature corpus.

No fixed disease register is required. Free-text disease names are matched to TCIA collections and GDC projects by keywords (and optionally by a free-tier LLM).

---

## What it can do

| Capability | Description |
|---|---|
| **Question → cohort** | Parse a natural-language research question into a scoped data request |
| **Paired matching** | Find patient IDs present in **both** TCIA (images) and GDC (genomics) |
| **Live or demo data** | `--catalog live` uses public TCIA/GDC APIs; `demo` uses synthetic in-memory data |
| **Radiomics + mutations** | Imaging features (CT/MR) and per-gene mutation flags on the paired cohort |
| **Statistics** | Pearson associations, FDR `q`-values, optional classifier metrics |
| **Literature labels** | Marks associations as supported / unverified / contradicted (built-in corpus) |
| **Optional LLM** | Orchestrator + DataMatcher can use Gemini/Groq free tiers for disease/source selection |

**Agents (deterministic by default):** Orchestrator → Data Matcher → Imaging + Genomics → Statistical/Critical → Literature → Discovery loop.

---

## Requirements

- **Python 3.10+**
- Network access for `--catalog live` (TCIA + GDC)
- Optional: GPU + PyTorch for TotalSegmentator-based segmentation
- Optional: `GEMINI_API_KEY` or `GROQ_API_KEY` for `--llm`

---

## Install

```bash
# Clone / enter the repo
cd AgenticRadioGen

# Recommended: virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS/WSL:
source .venv/bin/activate

# Core dependencies
pip install -r requirements.txt

# Editable install (so `python -m agentic_radiogen` works from anywhere)
pip install -e .

# Optional: tests
pip install -r requirements-dev.txt

# Optional: GPU imaging extras
pip install -r requirements-imaging-gpu.txt
```

Equivalent via `pyproject.toml`:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[imaging-gpu]"
```

Segmentation-model checkpoints are stored under the project's
`data_cache/seg_models/` directory by default (for example,
`data_cache/seg_models/nnunet_msd_lung/`). In WSL this is the same project
directory mounted under `/mnt/<drive>/...`, rather than the separate
`/home/<user>/.cache/agentic_radiogen/` tree. Existing model-specific cache
environment variables can still override the default location.

---

## Quick start (demo, no download)

Runs offline against a synthetic lung/breast catalog:

```bash
python -m agentic_radiogen \
  --stage 4 \
  --catalog demo \
  --question "Which imaging features are associated with EGFR mutations and survival in lung cancer?" \
  --out outputs/stage4_demo.json
```

---

## Live TCIA ∩ GDC run

Fetches **question-scoped** metadata (and optionally DICOM) for paired patients only — not full archives.

```bash
python -m agentic_radiogen \
  --stage 4 \
  --catalog live \
  --approve-download \
  --max-patients 100 \
  --question "Which imaging features are associated with specific genomic alterations in breast cancer?" \
  --out outputs/stage4_live.json
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--approve-download` | Required for live fetch (human gate) |
| `--no-dicom` | Metadata/mutations only (skip DICOM + radiomics) |
| `--max-patients N` | Cap after full i&g intersection |
| `--patient-id ID` | Restrict to one patient (or comma-separated IDs) |
| `--seg-model ID` | Force segmentation model (skip agent ranking), e.g. `nnunet_msd_lung` |
| `--modality CT` / `MR` | Override default modality |
| `--tcga-project` / `--tcia-collection` | Force fixed GDC/TCIA sources |
| `--genes` | Gene panel: `all` (default), `auto` (literature), one symbol (`TP53`), or comma list (`EGFR,KRAS`) |
| `--list-diseases` | Example disease phrases |
| `--no-llm` | Force rule-based orchestrator/matcher |
| `--llm` | Force free-tier LLM for planning + source matching |

### Optional nnU-Net Task998 lung CT model

The package supports `nnunet_task998_lungct`, an nnU-Net v1 lung tumor CT model.
Unlike Task006, its weights are **not downloaded automatically**. Use Linux or
WSL2 for the nnU-Net v1 environment.

1. Create a separate Python 3.10 environment and install this package, PyTorch,
   and nnU-Net v1:

   ```bash
   cd /mnt/g/Research/Projects/AgenticRadioGen
   python3.10 -m venv .venv-task998
   source .venv-task998/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "numpy==1.26.4"
   python -m pip install -e .
   ```

   Install a PyTorch build suitable for your CPU/CUDA setup using the
   [official PyTorch instructions](https://pytorch.org/get-started/locally/),
   then install nnU-Net v1 and NIfTI support:

   ```bash
   python -m pip install "nnunet==1.7.1" nibabel
   ```

2. Sign in to Kaggle and download the model archive from
   [CT Lung Tumor Segmentation model](https://www.kaggle.com/models/dejankuhn/ct-lung-tumor-segmentation-model).
   Review Kaggle's current access and usage terms before using the weights.
   The archive contains `Task998_LungCTSegmentation` at its root. Extract it
   into the repository's `data_cache/seg_models/` directory, retaining its
   trainer folder and checkpoint files. For the archive currently present at
   that path, from the repository root run:

   ```bash
   tar -xzf data_cache/seg_models/ct-lung-tumor-segmentation-model-pytorch-v1-v1.tar.gz \
     -C data_cache/seg_models/
   ```

   The resulting layout should include:

   ```text
   data_cache/seg_models/Task998_LungCTSegmentation/
     nnUNetTrainerV2__nnUNetPlansv2.1/
       plans.pkl
       fold_0/model_final_checkpoint.model
       fold_0/model_final_checkpoint.model.pkl
   ```

   The runner searches recursively under `data_cache/seg_models/` by default.
   If you extracted the task elsewhere, set
   `AGENTIC_RADIOGEN_TASK998_MODEL_DIR` to the directory containing the task
   folder (or the task folder itself).

3. Activate `.venv-task998` and explicitly select the model:

   ```bash
   python -m agentic_radiogen --stage 2 --catalog live --approve-download \
     --seg-model nnunet_task998_lungct \
     --question "Segment lung tumor on CT"
   ```

The segmentation log records the selected model and mask path for use with
`scripts/visualize_segmentation.py`. Automatic installation of nnU-Net weights
is not attempted; if the expected checkpoint files are absent, the pipeline
reports where to install them.

### Optional LLM (free tier)

```bash
# WSL / Linux / macOS
export GEMINI_API_KEY="your-key"   # from https://aistudio.google.com/apikey

python -m agentic_radiogen --stage 4 --catalog live --approve-download --llm \
  --question "Which imaging features are associated with mutations in lung cancer?"
```

Default model: **Gemini 3.8 Flash**. Alternative: `GROQ_API_KEY` + `--llm-provider groq`.  
If the LLM times out, the pipeline falls back to keyword rules automatically.

---

## Stages

| Stage | What runs |
|---|---|
| **1** | Orchestrator + Data Matcher (preview / fetch paired IDs) |
| **2** | Imaging + Genomics specialists |
| **3** | Stats join + literature annotation (no loop) |
| **4** | Full discovery loop (default) |

```bash
python -m agentic_radiogen --stage 1 --catalog demo --question "..."
```

---

## Outputs

### Terminal summary

Printed by default: disease, catalog, matched sources (TCIA ∩ GDC), paired patient counts, top associations, literature notes, loop decision.

### JSON (`--out path.json`)

Full run payload, including:

- `question`, `request` — parsed question and data plan  
- `patient_table` — per-patient radiomics × mutation matrix  
- `associations` — imaging feature ~ gene associations (`r`, `p`, `q`, `n`)  
- `associations_by_gene`, `top_associations`  
- `metrics` — optional model metrics (e.g. AUROC)  
- `literature` — supported / unverified / contradicted labels  
- `intermediate` — match keywords, collections/projects, i&g counts  
- `directive` / `stopped` — loop outcome  

### CSV (sibling of `--out`)

If associations exist, a CSV is written next to the JSON, e.g.:

- `outputs/stage4_live.json`
- `outputs/stage4_live.csv` — one row per association  

### Optional stdout JSON

```bash
python -m agentic_radiogen --stage 4 --catalog demo --json --out outputs/run.json
```

---

## How matching works (live)

1. Infer disease phrase from the question (rules or LLM).  
2. Rank / select TCIA collections and GDC projects (keywords or LLM).  
3. Prefer **pairable** sources that share TCGA-style IDs (e.g. `TCGA-BRCA`, `TCGA-LUAD`).  
4. Cohort = patient IDs in **imaging ∩ genomics** (`i&g`).  
5. Apply `--max-patients` after the full intersection.

Non-TCGA imaging collections often cannot pair with GDC (no shared IDs), so they may rank high by keyword but contribute little to `i&g`.

---

## Segmentation models (force with `--seg-model`)

Common ids: `nnunet_msd_lung`, `monai_vista3d`, `monai_lung_nodule`, `lungtumormask`, `ts_lung`.

Example — one patient with NVIDIA/MONAI VISTA-3D (lung tumor class 23):

```bash
python -m agentic_radiogen \
  --stage 4 --catalog live --approve-download --store-segm \
  --patient-id TCGA-17-Z021 \
  --seg-model monai_vista3d \
  --question "Which imaging features are associated with mutations in lung cancer?" \
  --out outputs/z021_vista3d.json
```

### Manual setup: MONAI VISTA-3D (`monai_vista3d`)

First use can auto-download the bundle. For a controlled / offline install:

```bash
# 1) Dependencies (GPU PyTorch recommended; install torch for your CUDA first)
pip install 'monai[nibabel,fire]' huggingface_hub nibabel

# 2) Download the official MONAI bundle (either works)
python -m monai.bundle download vista3d \
  --bundle_dir ~/.cache/agentic_radiogen/monai_vista3d

# or from Hugging Face:
# huggingface-cli download MONAI/vista3d \
#   --local-dir ~/.cache/agentic_radiogen/monai_vista3d/vista3d

# 3) Optional: point the app at the bundle (folder that contains configs/ + models/)
export AGENTIC_RADIOGEN_VISTA3D_BUNDLE=~/.cache/agentic_radiogen/monai_vista3d/vista3d

# Optional overrides:
# export AGENTIC_RADIOGEN_VISTA3D_CACHE=~/.cache/agentic_radiogen/monai_vista3d
# export AGENTIC_RADIOGEN_VISTA3D_LABEL=23   # lung tumor (default); 26=hepatic, 24=pancreatic, 27=colon
```

On Windows / WSL with a project under `/mnt/...`, the default cache is
`~/.cache/agentic_radiogen/monai_vista3d` (avoids slow `/mnt` I/O).

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Project layout

```
agentic_radiogen/
  agents/          # Orchestrator, DataMatcher, Imaging, Genomics, Stats, Literature
  data/            # Demo + live catalogs, TCIA/GDC clients, disease matching
  llm/             # Optional free-tier Gemini/Groq clients
  imaging/         # DICOM I/O, radiomics
  pipeline/        # Stages + discovery loop
  schemas/         # Pydantic contracts
```

---

## Notes and limits

- Live runs need network access to TCIA and GDC; large DICOM downloads can take time.  
- Free LLM quotas apply (Gemini daily limits reset at midnight Pacific).  
- Literature annotation uses a small built-in paper corpus (not a live PubMed search).  
- This software is for research prototyping; validate cohorts and statistics before clinical use.

---

## License

Check the repository for license terms. Public TCIA/GDC data remain under their respective usage policies.
