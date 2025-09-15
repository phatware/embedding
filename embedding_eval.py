#!/usr/bin/env python
"""
RC-EmbedBench (Theory-Aligned) — model/dataset fit via RC quantities.

What it computes (for each task):
  • JL capacity tolerance:    eta_JL(m, N, δ_target)
  • Dataset capacity frontier: Nmax(η*)
  • Channel deviation:        delta_op (||E_m∘(Φ-id)||_op on span(E_mΣ))
  • Information mapping band: C_low, C_high, C_ratio, BTI, DataFit
  • JS calibration check:     JS_pred_low/high = C_* * delta_op^2 vs empirical JS_mean
  • (Paraphrase only) AUC_cos and AUC_-JS for discrimination

Usage:
  python embedding_eval.py \
      --model text-embedding-3-large \
      --para_dataset mrpc --para_size 800 --balance True \
      --qa_dataset hotpot_qa --qa_size 400 \
      --eta_targets 0.15,0.10 --delta_target 1e-2
"""

import os, json, math, argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import sys
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine as cosine_distance
from dotenv import load_dotenv
from tqdm import tqdm
from embedding import get_embedding  # user-provided embedding function

# -----------------------------
# Config & constants
# -----------------------------
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EPS = 1e-12
H_DEFAULT = 64
# JL_CONSTANT provenance: Using m >= 8 ln(N/√δ) / η^2 form (tight up to constant factors) consistent with variants in Dasgupta & Gupta (2003) and standard JL lemma analyses.
JL_CONSTANT = 8.0

# -----------------------------
# Math helpers
# -----------------------------
def unit_rows(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return row-wise L2 unit-normalized copy of X (safe against zero / NaN).

    Any non-finite values are first replaced with 0. This ensures downstream
    dot products stay in [-1,1] (since both operands are unit), removing any
    need for per-coordinate clipping later.
    """
    X = np.asarray(X, dtype=float)
    # Replace NaN/Inf with 0 to avoid propagation of invalids
    mask_bad = ~np.isfinite(X)
    if np.any(mask_bad):
        X = X.copy()
        X[mask_bad] = 0.0
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, 1.0, n)  # avoid divide by ~0
    return X / n

def softmax_logits(z: np.ndarray, alpha: float) -> np.ndarray:
    """Stable softmax over z scaled by alpha.

    Assumes z is O(1); alpha is capped elsewhere. We still guard against
    accidental overflow by clipping the centered logits to a reasonable range.
    """
    x = alpha * z
    # center for stability
    x = x - np.max(x)
    # (rare) guard: clip extreme values if alpha*|z| unexpectedly large
    # if np.any(x > 80) or np.any(x < -80):  # exp(80) < 6.0e34 still finite
    #     x = np.clip(x, -80, 80)
    e = np.exp(x)
    S = e / max(np.sum(e), EPS)
    return S

def js_divergence_base2(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p/np.sum(p)
    q = q/np.sum(q)
    m = 0.5*(p+q)

    def kl(a, b):
        return np.sum(a*(np.log(a)-np.log(b))) / np.log(2.0)

    return 0.5*kl(p, m) + 0.5*kl(q, m)

def make_spherical_V(d: int, H: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((H, d))
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + EPS)
    return M  # (H,d) rows unit

def auto_choose_alpha(E_unit: np.ndarray, V: np.ndarray) -> float:
    """Heuristic scaling: α ≈ 1 / median std(V e).

    With e and rows of V both unit-normalized, each coordinate of z = V e lies
    in [-1,1], so std ≤ 1. This keeps α in a moderate range. We still cap α to
    100 for robustness (exp(±100) stays finite in float64).
    """
    if V.size == 0:
        print("auto_choose_alpha: empty probe V; using alpha=1.0")
        return 1.0
    Zstd: List[float] = []
    sample = E_unit[: min(4096, len(E_unit))]
    for e in sample:
        # No clipping necessary: dot products are already bounded in [-1,1].
        z = safe_proj(V, e)
        s = float(np.nanstd(z))
        if s > 0 and np.isfinite(s):
            Zstd.append(s)
    med = np.median(Zstd) if Zstd else 1.0
    if not np.isfinite(med) or med < 1e-6:
        med = 1.0
    return float(min(100.0, 1.0 / med))

def auto_choose_mu(H: int, c: float = 0.025) -> float:
    return max(1e-4, c / max(H, 1))

def belief_probs(e_unit: np.ndarray, V: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    """Return a smoothed probability vector derived from projection scores.

    Inputs:
        e_unit: unit-normalized embedding row (||e||_2=1)
        V: probe frame with unit-normalized rows
        alpha: softmax scale (chosen adaptively)
        mu: floor probability (prevents exact zeros)

    Numerical notes:
        - Because rows of V and e are unit, z ∈ [-1,1]^H, avoiding overflow.
        - No coordinate clipping is done (previous clip to [-1,1] was redundant).
    """
    z = safe_proj(V, e_unit)  # bounded & sanitized
    p = softmax_logits(z, alpha=alpha)
    p = np.clip(p, mu, 1.0)
    p = p / max(np.sum(p), EPS)
    return p

# ==== Hypothesis-set helpers ====

def build_hypothesis_set(E_unit: np.ndarray, sim_thresh: float = 0.90, max_hyp: int = 64, seed: int = 0) -> np.ndarray:
    """Greedy build of a hypothesis set (rows unit) with diversity cutoff.

    Adds a vector if its cosine similarity to all existing hypotheses is below
    sim_thresh. Stops after max_hyp. If E_unit is empty returns a (1,1) zero.
    """
    if E_unit.size == 0:
        return np.zeros((1,1), dtype=float)
    Hset: List[np.ndarray] = []
    # Shuffle to reduce order bias
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(E_unit))
    for e in E_unit[idx]:
        if not Hset:
            Hset.append(e)
        else:
            sims = [float(np.dot(e, h)) for h in Hset]
            if max(sims) < sim_thresh:
                Hset.append(e)
        if len(Hset) >= max_hyp:
            break
    return np.vstack(Hset) if Hset else np.zeros((1,1), dtype=float)

def belief_probs_hyp(e_unit: np.ndarray, Hset: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    """Probability over hypothesis set using softmax of dot products."""
    z = safe_proj(Hset, e_unit)
    p = softmax_logits(z, alpha=alpha)
    p = np.clip(p, mu, 1.0)
    p = p / max(np.sum(p), EPS)
    return p

def safe_proj(V: np.ndarray, e_unit: np.ndarray) -> np.ndarray:
    """Compute V @ e_unit with strong sanitization.

    Replaces any NaN/Inf in inputs with 0 and post-processes the result to be
    finite. This avoids sporadic low-level BLAS warnings (observed as divide by
    zero / overflow in matmul) that can arise if upstream embeddings contain
    pathological magnitudes before normalization.
    """
    if not np.all(np.isfinite(e_unit)):
        e_unit = np.nan_to_num(e_unit, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(V)):
        V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    # If e_unit norm drifts >1 (shouldn't), renormalize
    n = np.linalg.norm(e_unit)
    if n > 1.000001 or (n < 0.999 and n > 0):  # tolerate tiny float jitter
        e_unit = e_unit / n
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        z = V @ e_unit
    if not np.all(np.isfinite(z)):
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    # Final bounding (keeps downstream std estimation predictable)
    if z.size:
        z = np.clip(z, -1.0, 1.0)
    return z

def bti_ga(C_low, C_high, eps=EPS):
    Cl = max(C_low, eps); Ch = max(C_high, eps)
    return (2.0 * (Cl*Ch)**0.5) / (Cl + Ch)

def fit_para(BTI_ga, eta_JL, delta_op, CCS, k=2.0, eps=EPS):
    CCS = float(np.clip(CCS, 0.0, 1.0))
    denom = eta_JL * (1.0 + delta_op) + eps
    x = k * (BTI_ga / denom) * CCS
    return float(1.0 - np.exp(-x))

def fit_qa(BTI_ga, eta_JL, delta_op, CCS, cos_mean, k=2.0, eps=EPS):
    # map cos to [0,1] if it may be <0:
    c = EPS if cos_mean < EPS else cos_mean
    c = float(np.clip(c, 0.0, 1.0))
    denom = eta_JL * (1.0 + delta_op) + eps
    x = k * (BTI_ga / denom) * CCS * (1.0 - c)
    return float(1.0 - np.exp(-x))

def robust_band(js: np.ndarray, delta2: np.ndarray, qlo=0.05, qhi=0.95) -> Tuple[float,float,float,float]:
    mask = (delta2 > EPS) & np.isfinite(js)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = js[mask] / delta2[mask]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    C_low  = float(np.quantile(r, qlo))
    C_high = float(np.quantile(r, qhi))
    C_ratio = C_high / max(C_low, EPS)
    # BTI = 2.0 * C_low / (C_high + C_low)
    BTI = bti_ga(C_low, C_high)
    return C_low, C_high, C_ratio, BTI

def channel_correlation_score(js: np.ndarray, delta2: np.ndarray) -> float:
    """Channel Correlation Score (CCS): Pearson r(js, delta2).

    Interprets JS divergence as information mismatch and delta2 as squared
    channel distortion. High positive correlation indicates that *larger*
    geometric deviations (delta2) systematically induce larger information
    divergence (js) — a desirable monotonic alignment property.

    Returns NaN if insufficient variance or < 3 valid points.
    """
    if js is None or delta2 is None:
        return float("nan")
    js = np.asarray(js, dtype=float)
    delta2 = np.asarray(delta2, dtype=float)
    mask = np.isfinite(js) & np.isfinite(delta2)
    if not np.any(mask):
        return float("nan")
    x = delta2[mask]
    y = js[mask]
    # Need variability and at least 3 samples
    if x.size < 3:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    r = np.corrcoef(x, y)[0,1]
    return float(r) if np.isfinite(r) else float("nan")

def estimate_delta_operator(E: np.ndarray, Eprime: np.ndarray, lam: float = 1e-6) -> float:
    """Legacy ridge LS spectral norm estimator (kept for backward compat)."""
    if E.shape != Eprime.shape or E.shape[0] < 2:
        return float("nan")
    def _sanitize(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        if not np.all(np.isfinite(M)):
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        rn = np.linalg.norm(M, axis=1, keepdims=True)
        cap = 1e3
        mask = rn > cap
        if np.any(mask):
            M[mask[:,0]] = M[mask[:,0]] * (cap / rn[mask])
        return M
    E = _sanitize(E); Eprime = _sanitize(Eprime)
    X = E - E.mean(axis=0, keepdims=True)
    D = Eprime - E
    Y = D - D.mean(axis=0, keepdims=True)
    if np.linalg.norm(X) < 1e-9 or np.linalg.norm(Y) < 1e-9:
        return 0.0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        XtX = X.T @ X
    if not np.all(np.isfinite(XtX)):
        XtX = np.nan_to_num(XtX, nan=0.0, posinf=0.0, neginf=0.0)
    m = XtX.shape[0]
    try:
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            A = np.linalg.solve(XtX + lam * np.eye(m), X.T @ Y)
        svals = np.linalg.svd(A, compute_uv=False)
        return float(svals[0]) if svals.size else float("nan")
    except Exception:
        return float("nan")

def estimate_delta_operator_robust(E: np.ndarray, Eprime: np.ndarray, lam: Optional[float] = None, rank_tol: float = 1e-6) -> Tuple[float, float]:
    """Robust spectral norm estimate of channel deviation.

    Solves (X^T X + λ I) A = X^T Y in the *realized* subspace of X using SVD.
    Avoids inflating ||A|| when d >> n or X is ill-conditioned.

    Returns (spec_norm, residual_ratio) where residual_ratio = ||XA - Y||_F / (||Y||_F + eps).

    λ heuristic (if not provided): λ = (median(S)^2) * 1e-2 with floor 1e-6.
    Only singular directions with S_i / S_max >= rank_tol are used; others are ignored.
    """
    if E.shape != Eprime.shape or E.shape[0] < 2:
        return float("nan"), float("nan")
    eps = 1e-12
    def _sanitize(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        if not np.all(np.isfinite(M)):
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
        rn = np.linalg.norm(M, axis=1, keepdims=True)
        cap = 1e3
        mask = rn > cap
        if np.any(mask):
            M[mask[:,0]] = M[mask[:,0]] * (cap / rn[mask])
        return M
    E = _sanitize(E); Eprime = _sanitize(Eprime)
    X = E - E.mean(axis=0, keepdims=True)
    D = Eprime - E
    Y = D - D.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if np.linalg.norm(X) < 1e-9 or np.linalg.norm(Y) < 1e-9:
        return 0.0, 0.0
    # SVD of (n x d) matrix X
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except Exception:
        return float("nan"), float("nan")
    if S.size == 0:
        return float("nan"), float("nan")
    Smax = S[0]
    keep = S / (Smax + eps) >= rank_tol
    if not np.any(keep):
        return 0.0, 0.0
    U_r = U[:, keep]
    S_r = S[keep]
    Vt_r = Vt[keep, :]
    if lam is None:
        med = np.median(S_r)
        lam = max(1e-6, (med**2) * 1e-2)
    # Ridge filter factors f_i = S_i / (S_i^2 + lam)
    filt = S_r / (S_r**2 + lam)
    # Compute B = diag(filt) * U_r^T * Y  (shape r x d)
    B = (filt[:, None]) * (U_r.T @ Y)
    # A = V_r B  (V_r orthonormal => singular values(A) == singular values(B))
    try:
        sA = np.linalg.svd(B, compute_uv=False)
        spec = float(sA[0]) if sA.size else float("nan")
    except Exception:
        spec = float("nan")
    # Residual: X A - Y = U_r S_r Vt_r (V_r B) - Y = U_r S_r B - Y (since V_r^T V_r = I)
    # But X A = U_r S_r (Vt_r V_r) B = U_r S_r B (because Vt_r V_r = I_r). We never formed V_r.
    XA = U_r @ (S_r[:, None] * B)
    R = XA - Y
    resid_ratio = float(np.linalg.norm(R) / (np.linalg.norm(Y) + eps))
    # Scale adjustment: ensure spec not spuriously huge compared to relative change
    return spec, resid_ratio

# -----------------------------
# Load datasets
# -----------------------------

from datasets import load_dataset

def _extract_answer_sentence(context: str, a_text: str, a_start: int) -> str:
    if not context or a_start < 0:
        return a_text or ""
    # simple sentence split; swap in spacy or nltk if you want
    sents = [s.strip() for s in context.replace("\n", " ").split(". ") if s.strip()]
    # find sentence containing span
    end = a_start + len(a_text)
    # cheap char-based search: map cumulative lengths
    pos, acc = 0, []
    for s in sents:
        L = len(s) + (2 if not s.endswith(".") else 1)
        acc.append((pos, pos+L, s))
        pos += L
    for L, R, s in acc:
        if L <= a_start < R or L < end <= R:
            return f"Answer: {a_text}. In context: {s}"
    return f"Answer: {a_text}. In context: {context[max(0,a_start-80):min(len(context), a_start+len(a_text)+80)]}"

def _answer_window(context: str, a_text: str, a_start: int, window_chars: int = 160) -> str:
    if not context or a_start < 0:
        return a_text or ""
    L = max(0, a_start - window_chars)
    R = min(len(context), a_start + len(a_text) + window_chars)
    window = context[L:R].replace("\n", " ")
    return f"Answer: {a_text}. In context: {window}"

def load_qa_pairs(
    dataset: str,
    split: str = "train",
    answer_repr: str = "sentence",  # one of: "sentence"|"window"|"template"|"span"
    window_chars: int = 160
):
    qa = []
    dname = dataset.lower()

    if dname == "hotpot_qa":
        ds = load_dataset("hotpot_qa", "distractor")[split]
        # Use supporting sentences concatenation as answer context when available
        for r in ds:
            q = r.get("question", "")
            a = r.get("answer", "")
            ctxs = r.get("context", [])  # expected: list of [title, [sentences]]
            sup = r.get("supporting_facts", [])

            # Normalize supporting facts into mapping title -> list[int sentence indices]
            supports_by_title = {}
            for fact in sup:
                title = None; sent_idx = None
                if isinstance(fact, (list, tuple)):
                    if len(fact) >= 2:  # tolerate extra fields
                        title = fact[0]
                        sent_idx = fact[1]
                elif isinstance(fact, dict):
                    # Various possible key names
                    title = fact.get("title") or fact.get("page") or fact.get("doc")
                    sent_idx = fact.get("sent_id") or fact.get("sentence_id") or fact.get("idx")
                if title is None or sent_idx is None:
                    continue
                try:
                    si = int(sent_idx)
                except Exception:
                    continue
                supports_by_title.setdefault(title, []).append(si)

            ans_ctx = []
            # Build answer context from first two supporting titles (if available)
            for title, sidxs in list(supports_by_title.items())[:2]:
                # Find matching context entry
                ctx_entry = None
                for entry in ctxs:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        ctx_title, ctx_sents = entry[0], entry[1]
                    elif isinstance(entry, dict):
                        ctx_title = entry.get("title")
                        ctx_sents = entry.get("sentences") or entry.get("sents") or []
                    else:
                        continue
                    if ctx_title == title:
                        ctx_entry = (ctx_title, ctx_sents)
                        break
                if ctx_entry is None:
                    continue
                ctx_title, ctx_sents = ctx_entry
                # Collect chosen supporting sentences safely
                chosen = []
                for i in sorted(set(sidxs)):
                    if isinstance(ctx_sents, list) and 0 <= i < len(ctx_sents):
                        chosen.append(str(ctx_sents[i]))
                if chosen:
                    ans_ctx.append(" ".join(chosen))

            a_repr = f"Answer: {a}. In context: " + " ".join(ans_ctx) if ans_ctx else a
            qa.append((q, a_repr))

    elif dname == "trivia_qa":
        # Use the reading-comprehension split which includes evidence passages
        ds = load_dataset("trivia_qa", "rc")[split]
        for r in ds:
            q = r["question"]
            a_text = r.get("answer", {}).get("value", "") or ""
            # join available evidence docs; pick first non-empty
            # Some versions of the dataset surface entity_pages/evidence as dicts instead of lists.
            def _ensure_list(x):
                if x is None:
                    return []
                if isinstance(x, list):
                    return x
                if isinstance(x, dict):
                    # keep values (which are often dicts with text fields)
                    return list(x.values())
                return [x]
            entity_pages = _ensure_list(r.get("entity_pages", []))
            evidence = _ensure_list(r.get("evidence", []))
            contexts = entity_pages + evidence
            context = ""
            for c in contexts:
                if isinstance(c, dict):
                    context = c.get("text", "") or c.get("wiki_context", "") or ""
                elif isinstance(c, str):
                    context = c
                if context:
                    break
            if answer_repr == "sentence":
                # try to locate the answer string; if not found, fall back to window
                a_start = context.find(a_text) if a_text and context else -1
                a_repr = _extract_answer_sentence(context, a_text, a_start)
            elif answer_repr == "window":
                a_start = context.find(a_text) if a_text and context else -1
                a_repr = _answer_window(context, a_text, a_start, window_chars)
            elif answer_repr == "template":
                a_repr = f"Answer: {a_text}. In context: {context[:max(256,len(a_text)+80)]}"
            else:
                a_repr = a_text
            qa.append((q, a_repr))

    elif dname in ("qangaroo_wikihop", "wikihop"):
        ds = load_dataset("qangaroo", "wikihop")[split]
        for r in ds:
            q = r.get("query", "")
            a_text = r.get("answer", "")
            # supports is a list of passages; take first 1-2 to form multi-sentence context
            supports = r.get("supports", [])
            context = " ".join(supports[:2]) if supports else ""
            if answer_repr == "sentence":
                a_start = context.find(a_text) if a_text and context else -1
                a_repr = _extract_answer_sentence(context, a_text, a_start)
            elif answer_repr == "window":
                a_start = context.find(a_text) if a_text and context else -1
                a_repr = _answer_window(context, a_text, a_start, window_chars)
            elif answer_repr == "template":
                a_repr = f"Answer: {a_text}. In context: {context[:max(256,len(a_text)+80)]}"
            else:
                a_repr = a_text
            qa.append((q, a_repr))

    else:
        raise ValueError(f"Unsupported QA dataset: {dataset}")

    return qa

# -----------------------------
# Whitening utilities (for scale-comparable δ_op)
# -----------------------------

def _whiten_matrix(X: np.ndarray, tol: float = 1e-6, eps: float = 1e-6) -> np.ndarray:
    """Return a column-whitened version of X (rows = samples).

    Procedure: center rows, SVD (thin), keep singular components with S_i/S_0 >= tol,
    scale columns so covariance ~ I. Adds eps to variances for stability.
    """
    if X.size == 0:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return unit_rows(X)  # fallback
    if S.size == 0:
        return unit_rows(X)
    keep = S / (S[0] + eps) >= tol
    if not np.any(keep):
        return unit_rows(X)
    S_k = S[keep]
    Vt_k = Vt[keep, :]
    # Whitening transform: Xw = Xc @ V_k^T diag(1/sqrt(S_k^2/(n-1)+eps))
    n = X.shape[0]
    scales = 1.0 / np.sqrt((S_k**2)/(max(1, n-1)) + eps)
    Xw = Xc @ Vt_k.T * scales
    return Xw

def whiten_pair(E_Q: np.ndarray, E_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Independent whitening (legacy) — retained for fallback (not used for band/operator when joint whitening is preferable)."""
    return _whiten_matrix(E_Q), _whiten_matrix(E_A)

def joint_whiten_pair(E_Q: np.ndarray, E_A: np.ndarray, tol: float = 1e-6, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Joint whitening: stack Q and A, compute shared PCA basis, apply same transform.

    Ensures both sides lie in the same orthonormal coordinate system eliminating
    artificial alignment truncation. Returns (Qw, Aw).
    """
    if E_Q.size == 0 or E_A.size == 0:
        return E_Q, E_A
    Z = np.vstack([E_Q, E_A])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    except Exception:
        return unit_rows(E_Q), unit_rows(E_A)
    if S.size == 0:
        return unit_rows(E_Q), unit_rows(E_A)
    keep = S / (S[0] + eps) >= tol
    if not np.any(keep):
        return unit_rows(E_Q), unit_rows(E_A)
    S_k = S[keep]
    Vt_k = Vt[keep, :]
    n = Z.shape[0]
    scales = 1.0 / np.sqrt((S_k**2)/(max(1, n-1)) + eps)
    Zw = Zc @ Vt_k.T * scales
    Qw = Zw[:E_Q.shape[0]]
    Aw = Zw[E_Q.shape[0]:]
    return Qw, Aw

# -----------------------------
# Channel operator estimation orchestrator with fallbacks
# -----------------------------

def compute_delta_op(
    E_src: np.ndarray,
    E_tgt: np.ndarray,
    op_mode: str = "robust",
    space: str = "unit",
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, float, str]:
    """Unified wrapper to compute (delta_op, residual, note).

    Fallback order when NaNs encountered:
      1. Robust in requested space
      2. Robust in unit space (if different)
      3. Ridge in unit space
      4. Return (0.0, 1.0, 'fallback_zero')
    Sampling: if max_pairs provided and n > max_pairs, subsample rows.
    Note string records which path succeeded.
    """
    note = []
    if E_src.shape[0] != E_tgt.shape[0] or E_src.shape[0] < 2:
        return float("nan"), float("nan"), "invalid_shapes_n_samples"
    n = E_src.shape[0]
    if max_pairs and n > max_pairs:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_pairs, replace=False)
        E_src = E_src[idx]
        E_tgt = E_tgt[idx]
        note.append(f"subsampled_{max_pairs}")

    def prep(space_kind: str):
        if space_kind == "unit":
            return unit_rows(E_src), unit_rows(E_tgt)
        if space_kind == "whiten":
            return joint_whiten_pair(E_src, E_tgt)
        return E_src, E_tgt

    attempts: List[Tuple[str,str,str]] = []  # (mode, space, label)
    if op_mode == "robust":
        attempts.append(("robust", space, "primary"))
        if space != "unit":
            attempts.append(("robust", "unit", "unit_fallback"))
        attempts.append(("ridge", "unit", "ridge_unit"))
    else:  # ridge requested
        attempts.append(("ridge", space, "primary"))
        if space != "unit":
            attempts.append(("ridge", "unit", "ridge_unit"))
        attempts.append(("robust", "unit", "robust_unit"))

    for mode_try, space_try, label in attempts:
        X_op, Y_op = prep(space_try)
        if not np.all(np.isfinite(X_op)) or not np.all(np.isfinite(Y_op)):
            X_op = np.nan_to_num(X_op); Y_op = np.nan_to_num(Y_op)
        # Align feature dimensions if whitening produced different ranks
        if X_op.shape[1] != Y_op.shape[1]:
            d = min(X_op.shape[1], Y_op.shape[1])
            if d < 2:
                note.append(f"skip_{label}_low_common_dim")
                continue
            X_op = X_op[:, :d]
            Y_op = Y_op[:, :d]
            note.append(f"aligned_dim_{d}")
        if np.linalg.norm(X_op) < 1e-12 or np.linalg.norm(Y_op) < 1e-12:
            note.append(f"skip_{label}_zero_norm")
            continue
        if mode_try == "robust":
            d_op, d_res = estimate_delta_operator_robust(X_op, Y_op)
        else:
            d_op = estimate_delta_operator(X_op, Y_op)
            d_res = float("nan")
        if not np.isfinite(d_op):
            note.append(f"fail_{label}")
            continue
        if np.isnan(d_res):
            d_res = 0.0  # ridge path, residual not computed
        note.append(f"ok_{label}_{mode_try}_{space_try}")
        return d_op, d_res, ";".join(note)

    # All attempts failed
    note.append("all_failed")
    return 0.0, 1.0, ";".join(note)

def eta_jl(m: int, N: int, delta_target: float = 1e-2) -> float:
    # Invert m > JL_CONSTANT * ln(N/sqrt(δ)) / η^2  →  η = sqrt(JL_CONSTANT * ln(N/√δ) / m)
    N_eff = max(2, int(N))
    m_eff = max(1, int(m))
    return float(math.sqrt(JL_CONSTANT * math.log(N_eff / math.sqrt(delta_target)) / m_eff))

def Nmax_for_eta(eta_star: float, m: int, delta_target: float = 1e-2) -> float:
    # From m = JL_CONSTANT ln(N/√δ)/η^2 ⇒ ln(N/√δ) = m η^2 / JL_CONSTANT
    # Nmax(η*) = √δ * exp( (η*^2 * m) / JL_CONSTANT )
    return float(math.sqrt(delta_target) * math.exp((eta_star**2) * max(1, m) / JL_CONSTANT))

# -----------------------------
# Embedding I/O
# -----------------------------

def embed_texts(texts: List[str], model: str, pbar_desc: str = "Embedding") -> np.ndarray:
    """Fetch embeddings and build a (n,d) float array, padding with zeros.

    All invalid (NaN/Inf) values are zeroed. This keeps subsequent math stable.
    """
    vecs: List[np.ndarray] = []
    for t in tqdm(texts, desc=f"{pbar_desc} ({model})"):
        v = get_embedding(model, t)
        if v is None or getattr(v, "size", 0) == 0:
            # fail-safe zero vector with default dim
            dim = len(vecs[-1]) if vecs else 1536
            v = np.zeros(dim, dtype=float)
        arr = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(arr)):
            arr = arr.copy()
            arr[~np.isfinite(arr)] = 0.0
        # Guard against extreme magnitudes (shouldn't happen with sane models)
        max_abs = np.max(np.abs(arr)) if arr.size else 0.0
        if max_abs > 1e3:  # scale row down rather than hard clip to preserve direction
            arr = (arr / max_abs) * 1e3
        vecs.append(arr)
    d = max(len(v) for v in vecs)
    out = np.zeros((len(vecs), d), dtype=float)
    norm_cap = 1e3  # cap on row L2 norm to avoid giant dot products
    for i, v in enumerate(vecs):
        if v.size:
            n = np.linalg.norm(v)
            if n > norm_cap:
                v = (v / n) * norm_cap
        out[i, :len(v)] = v
    return out

# -----------------------------
# Paraphrase evaluation
# -----------------------------

def eval_paraphrase_auto(
    model: str,
    dataset: str = "mrpc",
    split="train",
    sample_size=800,
    seed=0,
    unit_delta: bool = False,
    balance: bool = False,
    H: int = H_DEFAULT,
    delta_target: float = 1e-2,
    eta_targets: List[float] = None,
    use_hyp: bool = False,
    hyp_sim: float = 0.90,
    hyp_max: int = 64,
    op_mode: str = "robust",
    delta_op_space: str = "unit",  # raw | unit | whiten
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if eta_targets is None:
        eta_targets = [0.15, 0.10]
    # Load pairs (text1, text2, label)
    if dataset == "mrpc":
        ds = load_dataset("glue","mrpc")[split]
        pairs = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds]
    elif dataset == "stsb":
        ds = load_dataset("glue", "stsb")[split]
        pairs = []
        for r in ds:
            a, b = r["sentence1"], r["sentence2"]
            score = float(r["label"])  # 0..5
            y = 1 if score >= 4.0 else 0
            pairs.append((a, b, y))
    elif dataset == "sick":
        ds = load_dataset("sick")[split]
        pairs = []
        for r in ds:
            a, b = r["sentence_A"], r["sentence_B"]
            entail = r["label"]  # 'ENTAILMENT' | 'CONTRADICTION' | 'NEUTRAL'
            rel = float(r.get("relatedness_score", 0.0))  # 1..5 (some configs include it)
            y = 1 if (entail == "ENTAILMENT" and rel >= 4.0) else 0
            pairs.append((a, b, y))
    else:
        raise ValueError("Unsupported paraphrase dataset.")

    # Balanced subsample if requested
    if sample_size and sample_size < len(pairs):
        if balance:
            pos = [p for p in pairs if p[2] == 1]
            neg = [p for p in pairs if p[2] == 0]
            k = min(len(pos), len(neg), sample_size // 2)
            pos_idx = rng.choice(len(pos), size=k, replace=False)
            neg_idx = rng.choice(len(neg), size=k, replace=False)
            pairs = [pos[int(i)] for i in pos_idx] + [neg[int(i)] for i in neg_idx]
            rng.shuffle(pairs)
        else:
            idx = rng.choice(len(pairs), size=sample_size, replace=False)
            pairs = [pairs[int(i)] for i in idx]

    texts = [t for (a,b,_) in pairs for t in (a,b)]
    E = embed_texts(texts, model, pbar_desc="Paraphrase")
    m = E.shape[1]
    E_unit = unit_rows(E)

    if use_hyp:
        Hset = build_hypothesis_set(E_unit, sim_thresh=hyp_sim, max_hyp=hyp_max, seed=seed)
        alpha = auto_choose_alpha(E_unit, Hset)
        mu = auto_choose_mu(Hset.shape[0])
    else:
        # Probe
        V = make_spherical_V(m, H, seed=0)
        alpha = auto_choose_alpha(E_unit, V)
        mu = auto_choose_mu(H)

    # Pairwise metrics
    cos, js, y = [], [], []
    # We'll build two delta2 variants: original (governed by unit_delta flag) and band-consistent
    delta2_orig = []
    E1_list, E2_list = [], []
    for i, (a,b,lbl) in enumerate(pairs):
        e1 = E[2*i]; e2 = E[2*i+1]
        u1 = E_unit[2*i]; u2 = E_unit[2*i+1]
        cos.append(1.0 - cosine_distance(u1, u2))
        if use_hyp:
            p1 = belief_probs_hyp(u1, Hset, alpha, mu)
            p2 = belief_probs_hyp(u2, Hset, alpha, mu)
        else:
            p1 = belief_probs(u1, V, alpha, mu)
            p2 = belief_probs(u2, V, alpha, mu)
        js.append(js_divergence_base2(p1, p2))
        if unit_delta:
            delta2_orig.append(float(np.sum((u2 - u1)**2)))
        else:
            delta2_orig.append(float(np.sum((e2 - e1)**2)))
        y.append(int(lbl))
        if lbl == 1:  # positives approximate Φ≈id
            E1_list.append(e1)
            E2_list.append(e2)

    cos = np.array(cos); js = np.array(js); delta2_orig = np.array(delta2_orig); y = np.array(y)
    auc_cos = roc_auc_score(y, cos) if len(set(y)) > 1 else float("nan")
    auc_negjs = roc_auc_score(y, -js) if len(set(y)) > 1 else float("nan")

    # Channel operator deviation (on positives only)
    delta_resid = float("nan")
    delta_note = ""
    if len(E1_list) >= 2:
        E_pos = np.vstack(E1_list)
        Epos_prime = np.vstack(E2_list)
        delta_op, delta_resid, delta_note = compute_delta_op(E_pos, Epos_prime, op_mode=op_mode, space=delta_op_space, rng=np.random.default_rng(seed))
    else:
        delta_op = float("nan")

    # Original behavior: band derived from original delta2 (unit or raw per unit_delta flag)
    C_low, C_high, C_ratio, BTI = robust_band(js, delta2_orig)
    # Operator-space delta (optional diagnostic) retained for predictions consistency if desired
    if delta_op_space == "unit":
        E_space = unit_rows(E)
    elif delta_op_space == "whiten":
        E_space = _whiten_matrix(E)
    else:
        E_space = E
    delta2_band = []
    for i in range(0, E_space.shape[0], 2):
        v1 = E_space[i]; v2 = E_space[i+1]
        delta2_band.append(float(np.sum((v2 - v1)**2)))
    delta2_band = np.array(delta2_band)

    JS_mean = float(np.mean(js)) if js.size else float("nan")
    JS_pred_low  = C_low  * (delta_op**2) if np.isfinite(delta_op) and np.isfinite(C_low)  else float("nan")
    JS_pred_high = C_high * (delta_op**2) if np.isfinite(delta_op) and np.isfinite(C_high) else float("nan")

    # Channel Correlation Score
    CCS = channel_correlation_score(js, delta2_orig)

    # JL capacity tolerance
    N = len(pairs) * 2  # number of embedded points used in this stage
    etaJL = eta_jl(m=m, N=N, delta_target=delta_target)
    cos_mean = np.mean(cos)

    fit = fit_para(BTI, etaJL, delta_op, CCS)

    # Compute Nmax for each eta target
    eta_results = {}
    for eta in eta_targets:
        eta_results[f"Nmax_eta_{eta}"] = Nmax_for_eta(eta, m=m, delta_target=delta_target)

    return {
        "model": model, "para_dataset": dataset, "n_pairs": len(pairs), "dim": m,
        "H": (Hset.shape[0] if use_hyp else H), "alpha": alpha, "mu": mu,
        "js_mode": ("hyp" if use_hyp else "probe"),
        # Discrimination
        "AUC_cos": float(auc_cos),
        "AUC_negJS": float(auc_negjs),
        # JS band & operator
        "JS_mean": JS_mean,
        "C_low": float(C_low),
        "C_high": float(C_high),
        "C_ratio": float(C_ratio),
        "BTI": float(BTI),
        "DataFit": float(fit),
        "delta_op": float(delta_op),
        "delta_op_resid": float(delta_resid),
        "delta_op_note": delta_note,
        "jl_constant": JL_CONSTANT,
        "JS_pred_low": float(JS_pred_low),
        "JS_pred_high": float(JS_pred_high),
        "CCS": float(CCS),
        # JL capacity
        "eta_JL": float(etaJL),
        **{k: float(v) for k, v in eta_results.items()},
        # Averages for intuition
        "cos_mean": float(np.mean(cos)) if cos.size else float("nan"),
    "delta_mean": float(np.mean(np.sqrt(delta2_orig))) if delta2_orig.size else float("nan"),
    "delta_mean_operator_space": float(np.mean(np.sqrt(delta2_band))) if delta2_band.size else float("nan"),
    }

# -----------------------------
# QA channel evaluation (Q → A)
# -----------------------------

def eval_qa_auto(
    model: str,
    dataset="hotpot_qa",
    split="train",
    sample_size=400,
    seed=0,
    unit_delta: bool = False,
    H: int = H_DEFAULT,
    delta_target: float = 1e-2,
    eta_targets: List[float] = None,
    use_hyp: bool = False,
    hyp_sim: float = 0.90,
    hyp_max: int = 64,
    op_mode: str = "robust",
    delta_op_space: str = "unit",  # raw | unit | whiten
    answer_repr: str = "span",      # span | sentence | template | window
    window_chars: int = 100,
    op_max_pairs: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if eta_targets is None:
        eta_targets = [0.15, 0.10]

    qa = load_qa_pairs(dataset, split=split, answer_repr=answer_repr, window_chars=window_chars)

    if sample_size and sample_size < len(qa):
        idx = rng.choice(len(qa), size=sample_size, replace=False)
        qa = [qa[int(i)] for i in idx]

    texts = [t for (q,a) in qa for t in (q,a)]
    E = embed_texts(texts, model, pbar_desc="QA")
    m = E.shape[1]
    E_unit = unit_rows(E)

    if use_hyp:
        Hset = build_hypothesis_set(E_unit, sim_thresh=hyp_sim, max_hyp=hyp_max)
        alpha = auto_choose_alpha(E_unit, Hset)
        mu = auto_choose_mu(Hset.shape[0])
    else:
        # Probe
        V = make_spherical_V(m, H, seed=0)
        alpha = auto_choose_alpha(E_unit, V)
        mu = auto_choose_mu(H)

    js, cos = [], []
    delta2_orig = []
    E_q_list, E_a_list = [], []
    for i in range(0, len(texts), 2):
        e_q, e_a = E[i], E[i+1]
        u_q, u_a = E_unit[i], E_unit[i+1]
        if use_hyp:
            p_q = belief_probs_hyp(u_q, Hset, alpha, mu)
            p_a = belief_probs_hyp(u_a, Hset, alpha, mu)
        else:
            p_q = belief_probs(u_q, V, alpha, mu)
            p_a = belief_probs(u_a, V, alpha, mu)
        js.append(js_divergence_base2(p_q, p_a))
        if unit_delta:
            delta2_orig.append(float(np.sum((u_a - u_q)**2)))
        else:
            delta2_orig.append(float(np.sum((e_a - e_q)**2)))
        cos.append(1.0 - cosine_distance(u_q, u_a))
        E_q_list.append(e_q); E_a_list.append(e_a)

    js = np.array(js); delta2_orig = np.array(delta2_orig); cos = np.array(cos)

    # Channel operator deviation (use all pairs; QA Φ is inherently lossy)
    E_Q = np.vstack(E_q_list); E_A = np.vstack(E_a_list)
    # Select representation space for operator estimation
    delta_op, delta_resid, delta_note = compute_delta_op(
        E_Q, E_A, op_mode=op_mode, space=delta_op_space,
        max_pairs=op_max_pairs, rng=rng, verbose=verbose)
    if verbose:
        print(f"[delta_op_debug] note={delta_note} value={delta_op:.4f} resid={delta_resid:.4f}")

    # Restore original behavior: use original delta2 (unit/raw) for band
    C_low, C_high, C_ratio, BTI = robust_band(js, delta2_orig)
    # Still compute operator-space deltas for diagnostics
    if delta_op_space == "unit":
        EQs = unit_rows(E_Q); EAs = unit_rows(E_A)
    elif delta_op_space == "whiten":
        EQs, EAs = joint_whiten_pair(E_Q, E_A)
    else:
        EQs, EAs = E_Q, E_A
    delta2_band = np.sum((EAs - EQs)**2, axis=1)
    JS_mean = float(np.mean(js)) if js.size else float("nan")
    JS_pred_low  = C_low  * (delta_op**2) if np.isfinite(delta_op) and np.isfinite(C_low)  else float("nan")
    JS_pred_high = C_high * (delta_op**2) if np.isfinite(delta_op) and np.isfinite(C_high) else float("nan")

    # Channel Correlation Score
    CCS = channel_correlation_score(js, delta2_orig)

    # JL capacity tolerance
    N = len(qa) * 2.0
    etaJL = eta_jl(m=m, N=N, delta_target=delta_target)

    cos_mean = float(np.mean(cos))
    fit = fit_qa(BTI, etaJL, delta_op, CCS, cos_mean)

    # Compute Nmax for each eta target
    eta_results = {}
    for eta in eta_targets:
        eta_results[f"Nmax_eta_{eta}"] = Nmax_for_eta(eta, m=m, delta_target=delta_target)

    return {
    "model": model, "qa_dataset": dataset, "n_pairs": len(qa), "dim": m,
        "H": (Hset.shape[0] if use_hyp else H), "alpha": alpha, "mu": mu,
        "js_mode": ("hyp" if use_hyp else "probe"),
        "answer_repr": answer_repr,
        "delta_op_space": delta_op_space,
        # JS band & operator
        "JS_mean": JS_mean,
        "C_low": float(C_low),
        "C_high": float(C_high),
        "C_ratio": float(C_ratio),
        "BTI": float(BTI),
        "DataFit": float(fit),
        "delta_op": float(delta_op),
        "delta_op_resid": float(delta_resid),
        "delta_op_note": delta_note,
        "jl_constant": JL_CONSTANT,
        "JS_pred_low": float(JS_pred_low),
        "JS_pred_high": float(JS_pred_high),
        "CCS": float(CCS),
        # JL capacity
        "eta_JL": float(etaJL),
        **{k: float(v) for k, v in eta_results.items()},
        # Intuition
        "cos_mean": cos_mean,
        "delta_mean": float(np.mean(np.sqrt(delta2_orig))) if delta2_orig.size else float("nan"),
        "delta_mean_operator_space": float(np.mean(np.sqrt(delta2_band))) if delta2_band.size else float("nan"),
    }

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=EMBEDDING_MODEL, help="Embedding model name or local path")

    ap.add_argument("--para_dataset", type=str, default=None, choices=["mrpc", "stsb"], help="Paraphrase dataset")
    ap.add_argument("--para_size", type=int, default=200, help="Number of paraphrase pairs to sample")
    ap.add_argument("--balance", action="store_true", help="Balance paraphrase positive/negative labels when sampling")

    ap.add_argument("--qa_dataset", type=str, default=None, choices=["hotpot_qa", "trivia_qa", "wikihop"], help="QA dataset")
    ap.add_argument("--qa_size", type=int, default=100, help="Number of QA pairs to sample")

    ap.add_argument("-s", "--seed", type=int, default=None, help="Global RNG seed for both tasks")

    ap.add_argument("-d", "--unit_delta", action="store_true", help="Use unit-normalized embeddings for delta distances (scale-invariant across models)")

    ap.add_argument("--H", type=int, default=H_DEFAULT, help="Probe frame size (kept fixed across models)")
    ap.add_argument("--delta_target", type=float, default=1e-2, help="Failure probability target for JL tolerance")
    ap.add_argument("--eta_targets", type=str, default="0.15,0.10", help="Comma list of η* for reporting Nmax(η*)")

    # Hypothesis-set options
    ap.add_argument("--hyp", type=float, default=0, help="Use hypothesis-set JSD instead of random probe frame with Cosine similarity threshold for adding new hypothesis vector (diversity)")
    ap.add_argument("--hyp-max", type=int, default=H_DEFAULT, help="Maximum number of hypothesis vectors")

    ap.add_argument("-o", "--out", type=str, default="rc_theory_eval_summary.json", help="Output JSON summary file.")
    ap.add_argument("--op_mode", type=str, default="robust", choices=["robust","ridge"], help="Operator deviation estimation mode (robust uses SVD subspace ridge).")
    ap.add_argument("--delta_space", type=str, default="unit", choices=["raw","unit","whiten"], help="Space used to estimate δ_op (raw embeddings, unit-normalized rows, or whitened).")
    ap.add_argument("--answer_repr", type=str, default="span", choices=["span","sentence","template","window"], help="Answer representation for QA datasets (SQuAD enhanced forms).")
    ap.add_argument("--window_size", type=int, default=100, help="Character window radius for window answer representation.")
    ap.add_argument("--max_pairs", type=int, default=None, help="Max pairs to use when fitting operator (subsample for stability/perf).")
    ap.add_argument("--verbose", action="store_true", help="Verbose diagnostics (prints delta_op fallback path).")
    args = ap.parse_args()

    if args.para_dataset is None and args.qa_dataset is None:
        print("Error: At least one of --para_dataset or --qa_dataset must be specified.")
        sys.exit(1)

    print("\n=== RC-EmbedBench (Theory-Aligned) ===")
    print(f"Model: {args.model}")

    # Parse eta_targets
    eta_targets = [float(x.strip()) for x in args.eta_targets.split(",") if x.strip()]

    # Run paraphrase
    seed = args.seed if args.seed is not None else 0
    if args.para_dataset is not None:
        res_p = eval_paraphrase_auto(
            args.model,
            dataset=args.para_dataset,
            sample_size=args.para_size,
            seed=seed,
            unit_delta=args.unit_delta,
            balance=args.balance,
            H=args.H,
            delta_target=args.delta_target,
            eta_targets=eta_targets,
            use_hyp=True if args.hyp > 0 else False,
            hyp_sim=args.hyp,
            hyp_max=args.hyp_max,
            op_mode=args.op_mode,
            delta_op_space=args.delta_space,
        )
    else:
        res_p = {}
    # Run QA

    if args.qa_dataset is not None:
        seed = (args.seed + 101) if args.seed is not None else 0
        res_q = eval_qa_auto(
            args.model,
            dataset=args.qa_dataset,
            sample_size=args.qa_size,
            seed=seed,
            unit_delta=args.unit_delta,
            H=args.H,
            delta_target=args.delta_target,
            eta_targets=eta_targets,
            use_hyp=True if args.hyp > 0 else False,
            hyp_sim=args.hyp,
            hyp_max=args.hyp_max,
            op_mode=args.op_mode,
            delta_op_space=args.delta_space,
            answer_repr=args.answer_repr,
            window_chars=args.window_size,
            op_max_pairs=args.max_pairs,
            verbose=args.verbose,
        )
    else:
        res_q = {}

    # Print succinct sections
    def fmt(v):
        if isinstance(v, float):
            return "nan" if not np.isfinite(v) else f"{v:.4f}"
        return str(v)

    def print_block(title: str, res: Dict[str,Any], keys: List[str]):
        print(f"\n[{title}]")
        for k in keys:
            if k in res:
                print(f"{k:>18}: {fmt(res[k])}")

    para_keys = [
        "model","para_dataset","n_pairs","dim","H","alpha","mu","js_mode",
        "AUC_cos","AUC_negJS",
        "JS_mean","C_low","C_high","C_ratio","BTI", "DataFit",
        "delta_op","delta_op_resid","delta_op_note","JS_pred_low","JS_pred_high","CCS",
        "eta_JL",
        "jl_constant",
    ]
    # Add dynamic eta results
    for eta in eta_targets:
        para_keys.append(f"Nmax_eta_{eta}")
    para_keys.extend(["cos_mean","delta_mean"])

    qa_keys = [
        "model","qa_dataset","n_pairs","dim","H","alpha","mu","js_mode","answer_repr","delta_op_space",
        "JS_mean","C_low","C_high","C_ratio","BTI", "DataFit",
        "delta_op","delta_op_resid","delta_op_note","JS_pred_low","JS_pred_high","CCS",
        "eta_JL",
        "jl_constant",
    ]
    # Add dynamic eta results
    for eta in eta_targets:
        qa_keys.append(f"Nmax_eta_{eta}")
    qa_keys.extend(["cos_mean","delta_mean"])
    print_block("Paraphrase (auto)", res_p, para_keys)
    print_block("QA channel (auto)", res_q, qa_keys)

    # Write summary JSON
    summary = {"paraphrase": res_p, "qa": res_q}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()

# python embedding_eval.py --qa_dataset wikihop --qa_size 50 --answer_repr window --window_size 120 --delta_space whiten --op_mode robust --max_pairs 500 --verbose --hyp 0.85 --hyp-max 64 --unit_delta --model ../llm-models/Qwen3-Embedding-8B --delta_target 1e-2 --seed 42
# python embedding_eval.py --para_dataset stsb --para_size 50 --answer_repr window --window_size 120 --delta_space whiten --op_mode robust --max_pairs 500 --verbose --hyp 0.85 --hyp-max 64 --unit_delta --model ../llm-models/Qwen3-Embedding-8B --delta_target 1e-2 --seed 42
