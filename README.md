---
dataset_name: s64-geometry-v1
pretty_name: "S64 Geometry Validation – The Conversational Coherence Region"
license: cc-by-4.0
language:
  - en
tags:
  - symbolic-ai
  - human-ai-interaction
  - embedding-geometry
  - semantic-space
  - conversation-dynamics
  - multi-model
task_categories:
  - other
papers:
  - title: "The Conversational Coherence Region: Geometry of Symbolic Meaning Across Embedding Models"
    url: https://www.aicoevolution.com/s64-geometry-paper
    doi: 10.5281/zenodo.18149380
repository: https://github.com/AICoevolution/paper02-coherence-region
---

# S64 Geometry Validation Dataset

> **Paper 02**: *The Conversational Coherence Region: Geometry of Symbolic Meaning Across Embedding Models*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18149380.svg)](https://doi.org/10.5281/zenodo.18149380)

This dataset accompanies the paper investigating how the S64 symbolic framework organizes semantically across 13 different embedding architectures, and how conversation dynamics reveal structured regions in semantic space.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **Architecture-Independent Structure** | Role centroids (from→to, through→result) show consistent angular relationships across all 13 backends (p < 0.001) |
| **Coherence Region** | Structured conversations occupy a distinct dynamical region: lower velocity, higher SGI, tighter symbol clustering |
| **Negative Cone-Diversity Correlation** | r = −0.88 [95% CI: −0.98, −0.55] between trajectory cone-ness and symbol diversity |
| **Large Effect Sizes** | Cohen's d = 1.15 for velocity differences between structured and unstructured conversations |

---

## Repository Structure

```
s64-geometry-v1/
│
├── README.md                           # This file
│
├── sweep/                              # Multi-backend symbol geometry analysis
│   ├── sweep_summary.json              # Summary: isotropy, clusters, role p-values (all backends)
│   ├── symbols.json                    # 180 S64/S128 symbol definitions with roles
│   ├── rosetta_dataset.json            # Consolidated 3D PCA positions for visualization
│   ├── run_meta.json                   # Sweep configuration and timestamp
│   │
│   └── per_backend/                    # Per-backend analysis (13 backends)
│       └── {backend}/
│           ├── embedding_meta.json     # Backend name, dimension, symbol count
│           └── symbol_geometry_analysis.json  # Angular stats, clusters, role geometry
│
├── conversations/                      # Conversation trajectory analysis
│   ├── traces_metrics.csv              # Primary data: 1 row per (conv × backend × mode)
│   ├── traces_metrics.json             # Same as above, JSON format
│   ├── cone_explanations.json          # Correlations: cone-ness vs diversity/entropy
│   ├── cross_model_agreement.json      # Jaccard overlap of top-k symbols across backends
│   └── run_meta.json                   # Analysis metadata
│   # Note: rosetta_conversations_pack.json (~100MB) available on request
│
├── dynamics/                           # Turn-level dynamics data
│   ├── manifold_dynamics.csv           # SGI and Velocity per turn, per conversation
│   └── statistical_analysis_results.json  # Bootstrap CIs, permutation tests, effect sizes
│
├── visualizations/                     # Interactive visualizations
│   ├── symbol_geometry_rosetta.html    # Multi-model 3D comparison viewer
│   └── alignment_hypersphere.html      # Conversation trajectory on hypersphere
│
└── scripts/                            # Analysis scripts (Python)
    ├── analyze_symbol_geometry.py      # Per-backend geometry analysis
    ├── analyze_conversation_geometry_pack.py  # Produces traces_metrics, cone_explanations
    ├── compute_statistical_tests.py    # Bootstrap, Mann-Whitney, permutation tests
    └── export_manifold_dynamics_csv.py # Exports SGI/velocity time series
```

---

## Backends Analyzed

| Backend | Dimension | Provider |
|---------|-----------|----------|
| `bge-m3` | 1024 | BAAI |
| `cohere-v3` | 1024 | Cohere |
| `e5-finetuned-v6` | 768 | Custom fine-tuned |
| `google` | 768 | Google |
| `jina-v3` | 1024 | Jina AI |
| `mistral-embed` | 1024 | Mistral AI |
| `nomic` | 768 | Nomic AI |
| `openai-3-large` | 3072 | OpenAI |
| `openai-3-small` | 1536 | OpenAI |
| `openai-ada-002` | 1536 | OpenAI |
| `qwen` | 1024 | Alibaba |
| `s128` | 768 | Custom (S128) |
| `voyage-large-2-instruct` | 1024 | Voyage AI |

---

## Conversations Analyzed

11 baseline conversations spanning structured therapeutic dialogue to unstructured exploration:

| ID | Type | Description |
|----|------|-------------|
| B01–B08 | Synthetic | Controlled baselines with ground-truth transformations |
| B09 | Naturalistic | Self-discovery dialogue (structured) |
| B10 | Naturalistic | AI interaction baseline |
| LC1 | Extended | Free-form exploratory conversation |

---

## Key Metrics Reference

| Metric | File | Description |
|--------|------|-------------|
| `highd_R` | traces_metrics.csv | 768D cone-ness: mean resultant length (0 = spread, 1 = tight) |
| `topk_unique_symbols` | traces_metrics.csv | Number of unique symbols activated across conversation |
| `topk_entropy_bits` | traces_metrics.csv | Shannon entropy of symbol usage |
| `step_angle_mean` | traces_metrics.csv | Mean angular distance between consecutive turns |
| `isotropy_score` | sweep_summary.json | How uniformly symbols are distributed (per backend) |
| `role_perm_p_close` | sweep_summary.json | P-value: role centroids unusually close |
| `role_perm_p_far` | sweep_summary.json | P-value: role centroids unusually far |
| `velocity` | manifold_dynamics.csv | Angular velocity per turn (degrees) |
| `sgi` | manifold_dynamics.csv | Semantic Grounding Index per turn |

---

## Quick Start

### Load conversation metrics (Python)

```python
import pandas as pd

# Load primary analysis data
df = pd.read_csv("conversations/traces_metrics.csv")

# Filter to centered mode (recommended)
df_centered = df[df["mode"] == "centered"]

# Correlation: cone-ness vs symbol diversity
print(df_centered[["highd_R", "topk_unique_symbols"]].corr())
```

### Load symbol geometry (Python)

```python
import json

# Load sweep summary
with open("sweep/sweep_summary.json") as f:
    summary = json.load(f)

# Check role geometry p-values across backends
for backend, data in summary["backends"].items():
    role_geo = data.get("role_geometry", {})
    p_close = role_geo.get("pvalue_unusually_close", {}).get("from-to")
    print(f"{backend}: from→to p = {p_close}")
```

### Interactive visualization

Open `visualizations/symbol_geometry_rosetta.html` in a browser to explore:
- 3D symbol lattice across all backends
- Role clustering visualization
- Conversation trajectory animation

---

## Statistical Methods

All statistical tests follow conservative, exploratory protocols:

- **Bootstrap CIs**: 10,000 resamples for correlation confidence intervals
- **Permutation tests**: 2,000 permutations for role geometry significance
- **Mann-Whitney U**: Non-parametric comparison of velocity distributions
- **Cohen's d**: Effect size for group comparisons
- **Stationarity testing**: Pearson correlation of velocity vs turn index

Results are reported with explicit sample-size caveats (N = 11 conversations).

---

## Citation

If you use this dataset, please cite:

```bibtex
@misc{jimenez2026coherenceregion,
  author = {Jiménez Sánchez, Juan Jacobo},
  title = {The Conversational Coherence Region: Geometry of Symbolic Meaning Across Embedding Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18149380},
  url = {https://doi.org/10.5281/zenodo.18149380}
}
```

Also cite the Semantic Grounding Index work:
- Marín, J. (2024). *Semantic Grounding Index for Large Language Models*. arXiv:2512.13771

---

## Related Work

- **Paper 01**: [S64: A Symbolic Framework for Human-AI Meaning Negotiation](https://www.aicoevolution.com/s64-paper)
- **S64 Dataset**: [s64-validation-v4](https://huggingface.co/datasets/AICoevolution/s64-validation-v4)

---

## Excluded Files

The following files are excluded from this repository due to HuggingFace size/binary limits:

- `rosetta_conversations_pack.json` (~100MB): Full embedding traces for all conversations
- `figures/*.png`: Paper figures (available in the [paper PDF](https://www.aicoevolution.com/s64-geometry-paper))
- `paper02-*.pdf`: Paper PDF (available at [aicoevolution.com/s64-geometry-paper](https://www.aicoevolution.com/s64-geometry-paper))

To request the complete dataset including large files, contact research@aicoevolution.com.

---

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Contact**: Juan Jacobo Jiménez Sánchez — [AICoevolution](https://www.aicoevolution.com)

