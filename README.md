## Repository for Systematic evaluation of protein language models highlights strengths and limitations
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22046470.svg)](https://doi.org/10.5281/zenodo.22046470)




### Data and Downloads
Datasets and supporting files are available through the [Zenodo repository](https://doi.org/10.5281/zenodo.22046470).

* Python pickle objects containing sequence embeddings used to train downstream models for predicting protein–protein interaction (PPI) affinity, subcellular localization (SCL), and protein solubility.
* FASTA source files used to construct the sequence databases for BLAST-based homolog searches.
* Human proteome selections generated using the MAPLE strategy with gamma = 1.0.

---

### Setup

```
conda env create -f environments/environment.yml
conda activate plm_env
```

This creates an env named `plm_env` with everything needed for every model family below (ESM, ProtBERT, ProtT5, CARP, ProGen2).

### Running an example

```
cd example
python example.py
```

This runs ESM-2 (8M) iterative masking on a short test sequence. First run needs internet access (to download model weights from HuggingFace into your local cache, `~/.cache/huggingface` by default); after that, cached weights are reused without needing the network again.

Expected output: use this to confirm your setup is working (per-position progress lines, then the results table. Exact values may differ slightly depending on hardware/library versions):
```
P155_HUMAN 0 / 17
...
P155_HUMAN 16 / 17
         score  token token_str  ...  site_entropy        s1  adj_score
name                             ...
M1M   0.657993     20         M  ...      2.298781  0.000000  0.664377
M1L   0.035542      4         L  ...      2.298781 -2.918481  0.035887
M1V   0.032935      7         V  ...      2.298781 -2.994646  0.033255
M1E   0.030001      9         E  ...      2.298781 -3.087956  0.030292
M1A   0.027771      5         A  ...      2.298781 -3.165202  0.028040

[5 rows x 9 columns]
```

### Supported models

Available through both `get_iterative_masked_df` (masking) and `get_embedding` (embeddings), using the same `model_key`:

| Family | `model_key` | Extra args needed |
|---|---|---|
| ESM (HuggingFace) | `esm_1b`, `esm_1v`, `esm_2_8M`, `esm_2_35M`, `esm_2_150M`, `esm_2_650M`, `esm_2_3B` | — |
| ESM (HuggingFace) | `esm_2_15B` | requires local cache (`huggingface_cache_dir`) — too large (~60GB) to auto-download, and needs a GPU with enough memory to actually run |
| ProtBERT (HuggingFace) | `protbert`, `protbert_bfd`* | — |
| ESM (direct `.pt` loading) | any key + a `.pt` path | `direct_loading_model_path` — supports both ESM-1 and ESM-2 architecture checkpoints |
| ProtT5 (HuggingFace) | `prott5_xl`, `prott5_xl_bfd`* | — |
| CARP | `carp_600k`*, `carp_38M`, `carp_76M`, `carp_640M` | optional `carp_model_path` to skip auto-download |
| ProGen2 | `progen2_medium`, `progen2_large`, `progen2_bfd` | `progen_model_dir`, `progen_code_dir` (see below) |

\* Not used in the manuscript's analysis but supported here.

---


### Masking usage

```python 
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

model_key = "esm_2_8M"
seq_results = get_iterative_masked_df(model_key, sequence, sequence_name)

```

ESM-1 or ESM-2 checkpoints can also be loaded directly from a downloaded `.pt` file instead of a `model_key`:

```python 
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"


pt_model_path = "./esm1_t6_43M_UR50S.pt"
model_key = "esm1_t6_43M_UR50S"
seq_results = get_iterative_masked_df(model_key, sequence, sequence_name, direct_loading_model_path = pt_model_path)

```

**CARP** — `model_key` in `carp_600k`, `carp_38M`, `carp_76M`, `carp_640M`. Weights auto-download from Zenodo the first time you use a given `model_key` (requires internet access). After that, they're cached locally and no network is needed. To skip the download entirely, pass `carp_model_path` pointing at a local `.pt` file instead.
```python
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

seq_results = get_iterative_masked_df("carp_600k", sequence, sequence_name)
```

**ProtT5** — `model_key` in `prott5_xl`, `prott5_xl_bfd`. Loaded from HuggingFace exactly like the ESM/ProtBERT models above (same caching behavior).
```python
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

seq_results = get_iterative_masked_df("prott5_xl", sequence, sequence_name)
```

**ProGen2** — `model_key` in `progen2_medium`, `progen2_large`, `progen2_bfd`. Unlike the others, ProGen2's weights and modeling code aren't distributed through this repo's normal channels, so two extra arguments are required: `progen_model_dir` (a local directory containing the downloaded `progen2-<variant>/` checkpoint) and `progen_code_dir` (a local clone of [salesforce/progen](https://github.com/salesforce/progen), i.e. the directory containing `progen/progen2/models/progen/modeling_progen.py`).

To set these up:

1. Clone the modeling code:
   ```
   git clone https://github.com/salesforce/progen.git
   ```
   This creates a `progen/` folder containing `progen2/models/progen/modeling_progen.py`. `progen_code_dir` is the directory you ran that command in (the parent of `progen/`), **not** the `progen/` folder itself.

2. Download the checkpoint(s) you want (`progen2-medium`, `progen2-large`, and/or `progen2-BFD90`) and the shared `tokenizer.json`, following the download instructions in that same repo. See its README for the current links/script. `progen_model_dir` is the directory holding the downloaded `progen2-<variant>/` folder(s) and `tokenizer.json` side by side:
   ```
   progen_model_dir/
     progen2-medium/
     tokenizer.json
   ```

```python
from plm_masking import get_iterative_masked_df

sequence_name = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

seq_results = get_iterative_masked_df(
    "progen2_medium", sequence, sequence_name,
    progen_model_dir="/path/to/your/progen2_models/",
    progen_code_dir="/path/to/your/clone/of/salesforce/progen/",
)
```
**Note:** ProGen2 is a causal (autoregressive) language model, not a masked language model — each position's score is computed from only the *preceding* sequence context, so it is not directly comparable to the bidirectional mask-infilling used by ESM, ProtBERT, ProtT5, and CARP, even though the output format is the same.

`seq_results` is a dataframe indexed by variant name (`<ref_aa><position><alt_aa>`, e.g. `M1L`) — see the [Expected output](#running-the-example) example above for a sample. Columns:

- `token` / `token_str` — the model alphabet index / amino acid for this row's variant
- `aa_pos`, `ref_aa` — the masked position and its wild-type amino acid
- `score` — raw probability of this token filling the masked position
- `adj_score` — `score` renormalized over just the 20 canonical amino acids
- `s1` — variant effect prediction score
- `site_entropy` — Shannon entropy of the position (same value for every row at that `aa_pos`)
- `gene` — the `sequence_name` you passed in

---

### Embedding extraction usage

Example code located within ``./example/example_embeddings.py``. Same `model_key` values and extra args as masking above, through an analogous `get_embedding` function.

```python
from plm_embeddings import get_embedding

id_ = "P155_HUMAN"
sequence = "MEMALMVAQTRKGKSVV"

model_key = "esm_2_8M"
embedding = get_embedding(model_key, sequence, id_)
```

Returns a dict with `mean_pooled` (mean-pooled residue embedding) and, for all model families except ProtT5 and CARP, `cls_pooled`.

