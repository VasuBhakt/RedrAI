"""
Computes semantic similarity between a job description and a candidate's
career narrative, using precomputed sentence embeddings.

This module does NOT compute embeddings itself; that's precompute_embeddings.py's
job (runs offline, untimed). This module only does the fast cosine-similarity
math during the timed ranking step.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity between two vectors, in [-1, 1] range
    (in practice, sentence-transformer embeddings are usually [0, 1] for
    semantically related text)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_semantic_scores(
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
) -> np.ndarray:
    """Vectorized cosine similarity for ALL candidates against the JD at once.

    candidate_embeddings: shape (N, D) — N candidates, D embedding dims
    jd_embedding: shape (D,) — single JD vector

    Returns: shape (N,) array of similarity scores, one per candidate.

    This is the version rank.py actually calls; doing this as a matrix
    operation instead of looping is what keeps 100K candidates inside the
    5-minute compute budget.
    """
    jd_norm = jd_embedding / np.linalg.norm(jd_embedding)
    cand_norms = candidate_embeddings / np.linalg.norm(
        candidate_embeddings, axis=1, keepdims=True
    )
    return cand_norms @ jd_norm


def normalize_to_unit_range(scores: np.ndarray) -> np.ndarray:
    """Rescales raw cosine similarities (often clustered in a narrow band,
    e.g. 0.15-0.45) to span the full [0, 1] range. This matters because if
    we don't rescale, semantic_score barely differentiates candidates when
    combined with other 0-1 scaled features in the weighted sum."""
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)