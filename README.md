# Choosing Meaning-Preserving Embeddings for RAG

This repository contains the implementation and evaluation framework accompanying our paper on principled embedding selection for Retrieval-Augmented Generation (RAG) systems within the [Recursive Consciousness](https://github.com/phatware/recursive-consciousness) (RC) theoretical framework.

## Overview

Most guides to embedding models for RAG emphasize empirical leaderboards and heuristics, leaving open the principled question of when an embedding is *large enough* and why one model should be preferred over another. This work provides theoretical foundations and practical tools for embedding selection based on meaning preservation guarantees. Full details are in the [paper](embedding.pdf).

### Key Contributions

1. **Theoretical Foundations**: Two-sided, testable information-theoretic bounds linking embedded channel deviation to belief disagreement in the RC framework
2. **Practical Metrics**:
   - **RC-EmbedScore**: Parameter-free evaluation combining paraphrase detection and semantic channel quality
   - **RAGFit**: Retrieval-specific evaluation measuring embedding suitability for RAG systems
   - **BTI (Bound Tightness Index)**: Novel metric quantifying embedding calibration quality
3. **Engineering Guidelines**: Concrete sizing recipes for embeddings and context windows based on application tolerances
4. **Comprehensive Evaluation**: Beyond size, catalog structural properties like isotropy, stability, and compositionality

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/phatware/embedding
cd embedding

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env to set your OPENAI_API_KEY
```

### Basic Usage

```bash
python embedding_eval.py -h
usage: embedding_eval.py [-h] [-m MODEL] [--para_dataset {mrpc,stsb}] [--para_size PARA_SIZE] [--balance] [--qa_dataset {hotpot_qa,trivia_qa,wikihop}]
                         [--qa_size QA_SIZE] [-s SEED] [-d] [--H H] [--delta_target DELTA_TARGET] [--eta_targets ETA_TARGETS] [--hyp HYP] [--hyp-max HYP_MAX]
                         [-o OUT] [--op_mode {robust,ridge}] [--delta_space {raw,unit,whiten}] [--answer_repr {span,sentence,template,window}]
                         [--window_size WINDOW_SIZE] [--max_pairs MAX_PAIRS] [--verbose]

options:
  -h, --help            show this help message and exit
  -m, --model MODEL     Embedding model name or local path
  --para_dataset {mrpc,stsb}
                        Paraphrase dataset
  --para_size PARA_SIZE
                        Number of paraphrase pairs to sample
  --balance             Balance paraphrase positive/negative labels when sampling
  --qa_dataset {hotpot_qa,trivia_qa,wikihop}
                        QA dataset
  --qa_size QA_SIZE     Number of QA pairs to sample
  -s, --seed SEED       Global RNG seed for both tasks
  -d, --unit_delta      Use unit-normalized embeddings for delta distances (scale-invariant across models)
  --H H                 Probe frame size (kept fixed across models)
  --delta_target DELTA_TARGET
                        Failure probability target for JL tolerance
  --eta_targets ETA_TARGETS
                        Comma list of η* for reporting Nmax(η*)
  --hyp HYP             Use hypothesis-set JSD instead of random probe frame with Cosine similarity threshold for adding new hypothesis vector (diversity)
  --hyp-max HYP_MAX     Maximum number of hypothesis vectors
  -o, --out OUT         Output JSON summary file.
  --op_mode {robust,ridge}
                        Operator deviation estimation mode (robust uses SVD subspace ridge).
  --delta_space {raw,unit,whiten}
                        Space used to estimate δ_op (raw embeddings, unit-normalized rows, or whitened).
  --answer_repr {span,sentence,template,window}
                        Answer representation for QA datasets (SQuAD enhanced forms).
  --window_size WINDOW_SIZE
                        Character window radius for window answer representation.
  --max_pairs MAX_PAIRS
                        Max pairs to use when fitting operator (subsample for stability/perf).
  --verbose             Verbose diagnostics (prints delta_op fallback path).
```

### Example Command

```bash
python embedding_eval.py \
    --model text-embedding-3-large \
    --para_dataset mrpc --para_size 800 --balance \
    --qa_dataset hotpot_qa --qa_size 400 \
    --eta_targets 0.15,0.10 --delta_target 1e-2 \
    --answer_repr window --window_size 120 --delta_space whiten \
    --op_mode robust --max_pairs 500 \
    --hyp 0.85 --hyp-max 64 --unit_delta
```

### Run Only One Task

Paraphrase only (MRPC, balanced):
```bash
python embedding_eval.py \
  --model text-embedding-3-small \
  --para_dataset mrpc --para_size 600 --balance \
  --eta_targets 0.15,0.10 --delta_target 1e-2 \
  --unit_delta --hyp 0.85 --hyp-max 64
```

QA only (HotpotQA, window answer representation):
```bash
python embedding_eval.py \
  --model text-embedding-3-large \
  --qa_dataset hotpot_qa --qa_size 300 \
  --answer_repr window --window_size 120 \
  --delta_space whiten --op_mode robust --max_pairs 400 \
  --eta_targets 0.15,0.10 --delta_target 1e-2 --unit_delta
```

### Using Local Embedding Models

Any local (SentenceTransformers) model can be passed by filesystem path; detection is automatic when the argument looks like a path (`./`, `../`, or absolute).

```bash
# Example with a local HuggingFace SentenceTransformer directory
python embedding_eval.py \
  --model ../llm-models/Qwen3-Embedding-8B \
  --para_dataset stsb --para_size 500 \
  --qa_dataset wikihop --qa_size 150 \
  --answer_repr window --window_size 140 \
  --delta_space whiten --op_mode robust --max_pairs 500 \
  --hyp 0.85 --hyp-max 64 --unit_delta
```

### Hypothesis-Set vs Probe Frame

Two ways to derive belief distributions for JS divergence:

1. Probe frame (default): random spherical `H` rows (basis-agnostic).
2. Hypothesis set (`--hyp <sim_thresh>`): greedily accumulate diverse embedding vectors whose pairwise cosine similarity remains below the threshold (e.g. `0.85`). This can yield a more data-adaptive belief space. Use `--hyp-max` to cap size.

To enable hypothesis-set mode set `--hyp` to the similarity threshold (0 disables it). The output field `js_mode` will be `hyp` when active.

### Key Output Metrics (JSON + Console)

By default, results are saved to `rc_theory_eval_summary.json` (you can set an alternative path from `--out`). Each top-level key (`paraphrase`, `qa`) holds a dictionary:

| Field | Meaning |
|-------|---------|
| `dim` | Embedding dimensionality `m` actually observed. |
| `eta_JL` | Johnson–Lindenstrauss tolerance estimate for current $(m, N, \delta)$. Smaller is better. |
| `Nmax_eta_*` | Capacity frontier: max number of points supportable at target distortion $\eta^*$. Larger $\to$ more headroom. |
| `delta_op` | Estimated channel operator deviation $\|\|E_m \circ (\Phi - id)\|\|_{op}$ (robust / ridge). Lower $\to$ more meaning preservation. |
| `delta_op_resid` | Relative residual of the robust fit (quality check). |
| `C_low`, `C_high` | 5%/95% robust band coefficients for $JS \approx C \times \delta^2$. |
| `C_ratio` | Band spread (`C_high / C_low`). Near 1 $\to$ tight calibration. |
| `BTI` | Bound Tightness Index (geometric-arithmetic proxy); closer to 1 $\to$ symmetric tight band. |
| `JS_pred_low/high` | Predicted JS for the operator using band edges. Empirical $JS_{mean}$ should fall between when calibration holds. |
| `CCS` | Channel Correlation Score: Pearson $r(JS, \delta^2)$; positive & high $\to$ monotonic distortion / info divergence relationship. |
| `DataFit` | RC-EmbedScore style composite fit (paraphrase adds discrimination; QA factors cosine alignment). Higher $\to$ better theory-data alignment. |
| `AUC_cos`, `AUC_negJS` | (Paraphrase) Discrimination scores: cosine similarity & negative-JS AUC for paraphrase vs non-paraphrase. |
| `cos_mean`, `delta_mean` | Intuitive averages (cosine similarity & mean $\delta$ over pairs). |

Interpretation Quick Rules:

* Good calibration: `BTI > 0.7`, `C_ratio < 3`, empirical `JS_mean` within `[JS_pred_low, JS_pred_high]`.
* Under-sized model: high `delta_op`, large `eta_JL`, low `Nmax_eta_*` simultaneously.
* Over-sized / diminishing returns: very low `delta_op` but `BTI` not improving versus smaller model.

### Choosing Flags

* Use `--unit_delta` when comparing models of different raw scale (normalizes distances before constructing δ²).
* Set `--delta_space whiten` to estimate operator deviation in a jointly whitened space (scale/comparison friendly) – especially helpful when models have anisotropic directions.
* Increase `--max_pairs` if you have plenty of positive paraphrase pairs or QA pairs and want a more stable operator estimate (diminishing returns after a few hundred).
* Lower `--H` to speed up runs (fewer probe rows) at the cost of noisier JS; raise for more stable bands.
* Adjust `--delta_target` and `--eta_targets` to explore application tolerances: tighter η* will shrink `Nmax_eta_*`.

### Programmatic Usage

You can import the evaluation routines directly:

```python
from embedding_eval import eval_paraphrase_auto, eval_qa_auto

res_p = eval_paraphrase_auto(
    model="text-embedding-3-small",
    dataset="mrpc",
    sample_size=400,
    unit_delta=True,
    hyp_sim=0.85,
    use_hyp=True,
)

res_q = eval_qa_auto(
    model="../llm-models/Qwen3-Embedding-8B",
    dataset="hotpot_qa",
    sample_size=200,
    answer_repr="window",
    delta_op_space="whiten",
)
```

### Reproducibility Tips

* Fix `--seed` to make subsampling and probe frame repeatable.
* For hypothesis sets, the greedy selection order uses the RNG permutation; same seed $\to$ same set.
* Store raw `rc_theory_eval_summary.json` for each model/config; downstream analysis notebooks (e.g. `process_results.ipynb`) can then aggregate across runs.

### Minimal Environment Variables

* `OPENAI_API_KEY`: required for OpenAI-hosted embedding models.
* `EMBEDDING_MODEL` (optional): default model used when `-m/--model` not provided.

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `openai.error.AuthenticationError` | Set `OPENAI_API_KEY` in `.env` or shell. |
| Extremely slow local model | Ensure you installed a GPU-enabled PyTorch build; reduce `--para_size` / `--qa_size`. |
| `delta_op_note` shows many fallbacks | Try `--delta_space unit` or reduce `--max_pairs`; inspect for NaNs in upstream embeddings. |
| `nan` metrics | Usually too few pairs after filtering; increase `--para_size` / `--qa_size` or disable balancing. |

### Citing

If you use these metrics or sizing recipes, please cite the accompanying paper (see `embedding.pdf` and replace DOI placeholder once assigned).

```
@inproceedings{rc-embedding-selection-2025,
  title     = {Choosing Meaning-Preserving Embeddings for RAG: From Infinite Banach to Finite Practical Vector Spaces},
  author    = {Stan Miasnikov},
  year      = {2025},
  note      = {Preprint; see repository for latest theory & benchmarks}
}
```
