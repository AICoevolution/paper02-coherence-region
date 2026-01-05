"""
Analyze the Geometry of S64 Symbol Embeddings

This script examines the 180 S64 symbols on the embedding hypersphere to look for:
1. Angular distribution patterns
2. Concentration vs a random hypersphere baseline (are symbols spread or "cap-like"?)
3. Semantic clustering structure (and "clumpiness" across cluster counts)
4. Role geometry (do S64 roles separate geometrically beyond chance?)
5. The "transducer" geometry between human symbols and AI vectors

Run from MirrorMind root:
    python _reports/Papers/paper02/validation/scripts/analyze_symbol_geometry.py

Requirements: numpy, scipy, matplotlib (optional)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Setup path
script_dir = Path(__file__).resolve().parent
validation_dir = script_dir.parent
paper02_dir = validation_dir.parent
reports_dir = paper02_dir.parent
mirror_mind_root = reports_dir.parent

# Add MirrorMind root to path (try multiple methods for robustness)
paths_to_try = [
    str(mirror_mind_root),
    str(Path.cwd()),  # Current working directory
    str(Path.cwd() / "MirrorMind"),  # If running from parent directory
]

for path in paths_to_try:
    if path not in sys.path:
        sys.path.insert(0, path)

# Verify we can import
try:
    from data.database import DatabaseService
except ImportError:
    # Try one more time with explicit path
    import os
    cwd = Path.cwd()
    if (cwd / "data" / "database.py").exists():
        sys.path.insert(0, str(cwd))
    elif (cwd / "MirrorMind" / "data" / "database.py").exists():
        sys.path.insert(0, str(cwd / "MirrorMind"))
    from data.database import DatabaseService

# Constants
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
GOLDEN_ANGLE_RAD = 2 * math.pi * (1 - 1/GOLDEN_RATIO)  # ≈ 2.399 rad ≈ 137.5°
GOLDEN_ANGLE_DEG = math.degrees(GOLDEN_ANGLE_RAD)

DEFAULT_RANDOM_BASELINE_SAMPLES = 25
DEFAULT_ROLE_PERMUTATION_SAMPLES = 2000


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def _sample_uniform_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points uniformly on the unit sphere in R^d.
    Uses Gaussian normalization: g ~ N(0, I), x = g / ||g||.
    """
    g = rng.normal(size=(n, d)).astype(np.float64)
    return _normalize_rows(g)


def mean_resultant_length(embeddings: np.ndarray) -> float:
    """
    Concentration proxy on the sphere: R̄ = ||Σ x_i|| / n for unit vectors x_i.
    - Uniform-ish on sphere: R̄ ~ 0
    - Concentrated ("cap-like"): R̄ closer to 1
    """
    x = _normalize_rows(embeddings.astype(np.float64))
    v = np.sum(x, axis=0)
    return float(np.linalg.norm(v) / max(len(x), 1))


def estimate_vmf_kappa(r_bar: float, d: int) -> Optional[float]:
    """
    Very rough κ estimate for a von Mises–Fisher distribution in d dimensions.
    This is used only as an interpretable *concentration scale*, not a claim the data is vMF.
    """
    if d < 3:
        return None
    if r_bar <= 1e-9:
        return 0.0
    if r_bar >= 0.999999:
        return float("inf")
    # Common approximation:
    # κ ≈ R̄*(d - R̄^2)/(1 - R̄^2)
    return float((r_bar * (d - r_bar**2)) / (1.0 - r_bar**2))


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compare_to_random_baseline(
    embeddings: np.ndarray,
    n_samples: int = DEFAULT_RANDOM_BASELINE_SAMPLES,
    seed: int = 1337,
) -> Dict:
    """
    Compare the observed symbol geometry to random points uniformly distributed on the same sphere.

    Returns:
      - angle stats for random samples (mean/std distribution)
      - KS test between observed and one random sample (as a simple distributional difference proxy)
      - concentration metrics (R̄ and κ estimate)
    """
    n, d = embeddings.shape
    rng = np.random.default_rng(seed)

    # Observed
    obs_angles_matrix, obs_angles_flat = compute_angular_distances(embeddings)
    obs_angle_mean = float(np.mean(obs_angles_flat))
    obs_angle_std = float(np.std(obs_angles_flat))
    obs_rbar = mean_resultant_length(embeddings)
    obs_kappa = estimate_vmf_kappa(obs_rbar, d)

    rand_angle_means = []
    rand_angle_stds = []
    rand_rbars = []
    rand_kappas = []

    # We'll also compute one representative random sample for a KS test against observed.
    ks_stat = None
    ks_pvalue = None

    for i in range(max(n_samples, 1)):
        r = _sample_uniform_sphere(n=n, d=d, rng=rng)
        _, rand_angles_flat = compute_angular_distances(r)
        rand_angle_means.append(float(np.mean(rand_angles_flat)))
        rand_angle_stds.append(float(np.std(rand_angles_flat)))
        rbar = mean_resultant_length(r)
        rand_rbars.append(float(rbar))
        kappa = estimate_vmf_kappa(rbar, d)
        rand_kappas.append(float(kappa) if kappa is not None and np.isfinite(kappa) else float("nan"))

        if i == 0:
            ks = stats.ks_2samp(obs_angles_flat, rand_angles_flat)
            ks_stat = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)

    rand_angle_means_arr = np.asarray(rand_angle_means, dtype=np.float64)
    rand_angle_stds_arr = np.asarray(rand_angle_stds, dtype=np.float64)
    rand_rbars_arr = np.asarray(rand_rbars, dtype=np.float64)
    rand_kappas_arr = np.asarray(rand_kappas, dtype=np.float64)

    # Effect sizes: how extreme is observed vs random distribution of summary metrics?
    def _percentile_of_score(arr: np.ndarray, score: float) -> float:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.mean(arr <= score) * 100.0)

    return {
        "n_symbols": int(n),
        "embedding_dim": int(d),
        "baseline_samples": int(n_samples),
        "seed": int(seed),
        "observed": {
            "pairwise_angle_mean_deg": obs_angle_mean,
            "pairwise_angle_std_deg": obs_angle_std,
            "mean_resultant_length_rbar": float(obs_rbar),
            "kappa_estimate": float(obs_kappa) if obs_kappa is not None else None,
        },
        "random_baseline": {
            "pairwise_angle_mean_deg": _summary_stats(rand_angle_means_arr),
            "pairwise_angle_std_deg": _summary_stats(rand_angle_stds_arr),
            "mean_resultant_length_rbar": _summary_stats(rand_rbars_arr),
            "kappa_estimate": _summary_stats(rand_kappas_arr),
            "percentile_of_observed": {
                "pairwise_angle_mean_deg": _percentile_of_score(rand_angle_means_arr, obs_angle_mean),
                "pairwise_angle_std_deg": _percentile_of_score(rand_angle_stds_arr, obs_angle_std),
                "mean_resultant_length_rbar": _percentile_of_score(rand_rbars_arr, obs_rbar),
            },
        },
        "distribution_tests": {
            "ks_test_observed_vs_random_sample0": {
                "statistic": ks_stat,
                "pvalue": ks_pvalue,
            }
        },
    }


def gini_coefficient(values: List[int]) -> float:
    """
    Gini coefficient for non-negative values.
    0 = perfectly even
    1 = maximally uneven
    """
    x = np.array(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini = (n+1 - 2 * Σ (cumx)/cumx[-1]) / n
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def cluster_sweep(symbols: List["SymbolData"], angles_matrix: np.ndarray, cluster_counts: List[int]) -> Dict:
    sweep = {}
    for k in cluster_counts:
        res = find_semantic_clusters(symbols, angles_matrix, n_clusters=int(k))
        sizes = res["cluster_sizes"]
        total = max(sum(sizes), 1)
        proportions = [s / total for s in sizes]
        inv_simpson = float(1.0 / np.sum(np.square(proportions))) if proportions else float("nan")
        sweep[str(k)] = {
            "cluster_sizes": sizes,
            "gini": gini_coefficient(sizes),
            "max_cluster_fraction": float(max(proportions) if proportions else float("nan")),
            "inv_simpson_effective_clusters": inv_simpson,
        }
    return sweep


def _role_members_unique(symbols: List["SymbolData"]) -> Dict[str, List[int]]:
    roles = {"from": [], "through": [], "to": [], "observation": [], "result": []}
    for i, sym in enumerate(symbols):
        for role, count in (sym.roles or {}).items():
            if role in roles and count and count > 0:
                roles[role].append(i)
    # Deduplicate while preserving order
    for r in list(roles.keys()):
        seen = set()
        dedup = []
        for idx in roles[r]:
            if idx in seen:
                continue
            seen.add(idx)
            dedup.append(idx)
        roles[r] = dedup
    return roles


def _centroid_unit(embeddings: np.ndarray, indices: List[int]) -> Optional[np.ndarray]:
    if not indices:
        return None
    x = _normalize_rows(embeddings[indices])
    c = np.mean(x, axis=0)
    norm = np.linalg.norm(c)
    if norm <= 0:
        return None
    return c / norm


def role_centroid_angles_unique(symbols: List["SymbolData"], embeddings: np.ndarray) -> Dict[str, float]:
    members = _role_members_unique(symbols)
    centroids: Dict[str, np.ndarray] = {}
    for role, idxs in members.items():
        c = _centroid_unit(embeddings, idxs)
        if c is not None:
            centroids[role] = c

    angles = {}
    roles = list(centroids.keys())
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            cos_sim = float(np.dot(centroids[roles[i]], centroids[roles[j]]))
            ang = float(math.degrees(math.acos(np.clip(cos_sim, -1, 1))))
            angles[f"{roles[i]}-{roles[j]}"] = ang
    return angles


def permutation_test_role_separation(
    symbols: List["SymbolData"],
    embeddings: np.ndarray,
    pairs_to_test: Optional[List[Tuple[str, str]]] = None,
    n_permutations: int = DEFAULT_ROLE_PERMUTATION_SAMPLES,
    seed: int = 1337,
    save_null_distribution_values: bool = False,
) -> Dict:
    """
    Test if role centroids are unusually *close* compared to random role assignments.

    Null: role membership is random (but role set sizes are preserved).
    We allow overlaps across roles (sampling is independent per role), since real roles are non-exclusive.
    """
    if pairs_to_test is None:
        pairs_to_test = [("from", "to"), ("through", "result"), ("observation", "result")]

    members = _role_members_unique(symbols)
    sizes = {r: len(idxs) for r, idxs in members.items()}
    n, d = embeddings.shape
    rng = np.random.default_rng(seed)

    # Observed angles
    obs_angles = {}
    obs_centroids = {r: _centroid_unit(embeddings, members[r]) for r in members.keys()}
    for a, b in pairs_to_test:
        if obs_centroids.get(a) is None or obs_centroids.get(b) is None:
            continue
        cos_sim = float(np.dot(obs_centroids[a], obs_centroids[b]))
        obs_angles[f"{a}-{b}"] = float(math.degrees(math.acos(np.clip(cos_sim, -1, 1))))

    # Permutation distribution
    dist = {k: [] for k in obs_angles.keys()}

    for _ in range(max(n_permutations, 1)):
        # Random role membership with preserved set sizes.
        rand_centroids = {}
        for role, k in sizes.items():
            if k <= 0:
                continue
            idxs = rng.choice(n, size=k, replace=False)
            c = _centroid_unit(embeddings, list(idxs))
            if c is not None:
                rand_centroids[role] = c

        for key in dist.keys():
            a, b = key.split("-", 1)
            if rand_centroids.get(a) is None or rand_centroids.get(b) is None:
                continue
            cos_sim = float(np.dot(rand_centroids[a], rand_centroids[b]))
            ang = float(math.degrees(math.acos(np.clip(cos_sim, -1, 1))))
            dist[key].append(ang)

    # p-value for "unusually close": P(null_angle <= observed_angle)
    pvals_close = {}
    pvals_far = {}
    for key, vals in dist.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            pvals_close[key] = None
            pvals_far[key] = None
            continue
        obs = float(obs_angles[key])
        p_close = float(np.mean(arr <= obs))
        p_far = float(np.mean(arr >= obs))
        pvals_close[key] = p_close
        pvals_far[key] = p_far

    return {
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "role_set_sizes_unique": sizes,
        "observed_angles_deg": obs_angles,
        "null_distribution_summary": {k: _summary_stats(np.asarray(v, dtype=np.float64)) for k, v in dist.items()},
        "pvalue_unusually_close": pvals_close,
        "pvalue_unusually_far": pvals_far,
        **(
            {"null_distribution_values_deg": {k: v for k, v in dist.items()}}
            if save_null_distribution_values
            else {}
        ),
    }


@dataclass
class SymbolData:
    symbol: str
    token: str
    embedding: List[float]
    roles: Dict[str, int]
    paths: Dict[str, str]


def fetch_symbols(backend: str = "s128") -> List[SymbolData]:
    """Fetch all symbol embeddings from the database."""
    db = DatabaseService()
    
    # Determine which column to use
    col_map = {
        # Explicit local columns (avoid ambiguity with legacy `embedding`)
        "s128": "embedding_s128",
        "local": "embedding_s128",
        "e5-finetuned-v6": "embedding_e5_finetuned_v6",
        "e5-v6": "embedding_e5_finetuned_v6",
        "e5": "embedding_e5_finetuned_v6",
        "openai": "embedding_openai",
        "openai-3-small": "embedding_openai_3_small",
        "openai-3-large": "embedding_openai_3_large",
        "cohere": "embedding_cohere",
        "nomic": "embedding_nomic",
        "bge-m3": "embedding_bge_m3",
        "google": "embedding_google",
    }
    col = col_map.get(backend, "embedding")
    
    query = f"""
        SELECT symbol, token, {col} as embedding, roles, paths
        FROM s64_symbol_embeddings
        WHERE {col} IS NOT NULL
        ORDER BY symbol
    """
    
    rows = db.execute_app_query(query, dict_cursor=True)
    
    symbols = []
    for row in rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            # Parse pgvector format
            emb = json.loads(emb.replace('{', '[').replace('}', ']'))
        
        roles = row.get("roles") or {}
        if isinstance(roles, str):
            roles = json.loads(roles)
        
        paths = row.get("paths") or {}
        if isinstance(paths, str):
            paths = json.loads(paths)
        
        symbols.append(SymbolData(
            symbol=row["symbol"],
            token=row["token"],
            embedding=emb,
            roles=roles,
            paths=paths,
        ))
    
    return symbols


def compute_angular_distances(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all pairwise angular distances (in degrees).
    
    Returns:
        angles_matrix: NxN matrix of angular distances
        angles_flat: flattened upper triangle (unique pairs)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms > 0, norms, 1)
    
    # Cosine similarity matrix
    cos_sim = normalized @ normalized.T
    
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    cos_sim = np.clip(cos_sim, -1, 1)
    
    # Convert to angles in degrees
    angles_matrix = np.degrees(np.arccos(cos_sim))
    
    # Get upper triangle (excluding diagonal)
    angles_flat = angles_matrix[np.triu_indices_from(angles_matrix, k=1)]
    
    return angles_matrix, angles_flat


def analyze_angular_distribution(angles_flat: np.ndarray) -> Dict:
    """Analyze the distribution of angular distances."""
    return {
        "min_angle": float(np.min(angles_flat)),
        "max_angle": float(np.max(angles_flat)),
        "mean_angle": float(np.mean(angles_flat)),
        "median_angle": float(np.median(angles_flat)),
        "std_angle": float(np.std(angles_flat)),
        "percentiles": {
            "5%": float(np.percentile(angles_flat, 5)),
            "25%": float(np.percentile(angles_flat, 25)),
            "50%": float(np.percentile(angles_flat, 50)),
            "75%": float(np.percentile(angles_flat, 75)),
            "95%": float(np.percentile(angles_flat, 95)),
        }
    }

def preprocess_embeddings(
    embeddings: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Preprocess embeddings before geometry analysis.

    Why this exists:
      Many modern embedding models are anisotropic: vectors have a strong shared mean direction,
      which makes *everything* look like it lies on a narrow cone. This can dominate angular stats.

    Modes:
      - "raw": unit-normalize only
      - "centered": subtract mean vector, then unit-normalize
      - "whitened": mean-center, then PCA-whiten (using all components), then unit-normalize
    """
    x = embeddings.astype(np.float64)
    if mode == "raw":
        return _normalize_rows(x)
    if mode == "centered":
        x = x - np.mean(x, axis=0, keepdims=True)
        return _normalize_rows(x)
    if mode == "whitened":
        x = x - np.mean(x, axis=0, keepdims=True)
        # PCA-whiten in the sample subspace
        # x = U S V^T; whitened = U * sqrt(n-1)
        # (equivalent to x @ V @ diag(1/S) * sqrt(n-1))
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        eps = 1e-12
        scale = math.sqrt(max(x.shape[0] - 1, 1))
        xw = (u * scale)  # implicitly divides by s (absorbed by u when using U)
        # NOTE: This produces an (n x r) representation; keep as-is for angular analysis.
        return _normalize_rows(xw)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def check_fibonacci_patterns(angles_flat: np.ndarray) -> Dict:
    """
    Look for Fibonacci/golden ratio patterns in the angular distribution.
    
    Key angles to check:
    - Golden angle: 137.5° (optimal packing)
    - Complementary: 222.5° (360 - 137.5)
    - Half golden: 68.75°
    - Fibonacci ratios in angle distribution
    """
    results = {}
    
    # Check for peaks near golden angle
    golden_angle = GOLDEN_ANGLE_DEG
    tolerance = 5.0  # degrees
    
    near_golden = np.sum(np.abs(angles_flat - golden_angle) < tolerance)
    near_half_golden = np.sum(np.abs(angles_flat - golden_angle/2) < tolerance)
    near_complement = np.sum(np.abs(angles_flat - (360 - golden_angle)) < tolerance)
    
    total_pairs = len(angles_flat)
    
    results["golden_angle"] = {
        "target": golden_angle,
        "tolerance": tolerance,
        "count_near": int(near_golden),
        "percentage": float(near_golden / total_pairs * 100) if total_pairs > 0 else 0,
    }
    
    results["half_golden"] = {
        "target": golden_angle / 2,
        "count_near": int(near_half_golden),
        "percentage": float(near_half_golden / total_pairs * 100) if total_pairs > 0 else 0,
    }
    
    # Histogram to find peaks
    bins = np.arange(0, 185, 5)  # 0-180 degrees in 5-degree bins
    hist, bin_edges = np.histogram(angles_flat, bins=bins)
    
    # Find peaks
    peak_indices = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peak_indices.append(i)
    
    peaks = [(bin_edges[i] + 2.5, int(hist[i])) for i in peak_indices]
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    results["distribution_peaks"] = peaks[:10]  # Top 10 peaks
    
    # Check if peak ratios follow Fibonacci
    if len(peaks) >= 2:
        ratios = []
        for i in range(len(peaks) - 1):
            ratio = peaks[i][0] / peaks[i+1][0] if peaks[i+1][0] > 0 else 0
            ratios.append(ratio)
        
        # How close are ratios to golden ratio or its inverse?
        golden_similarity = [
            abs(r - GOLDEN_RATIO) if r > 1 else abs(1/r - GOLDEN_RATIO) if r > 0 else float('inf')
            for r in ratios
        ]
        results["peak_ratios"] = list(zip([p[0] for p in peaks[:-1]], ratios, golden_similarity))
    
    return results


def find_semantic_clusters(
    symbols: List[SymbolData],
    angles_matrix: np.ndarray,
    n_clusters: int = 12
) -> Dict:
    """
    Cluster symbols by angular proximity and analyze cluster structure.
    
    We use 12 clusters as a hypothesis (12 = Fibonacci number, zodiac, hours).
    """
    # Hierarchical clustering using angular distance
    # Ensure diagonal is zero (distance from symbol to itself)
    angles_matrix = angles_matrix.copy()
    np.fill_diagonal(angles_matrix, 0.0)
    
    # Convert square matrix to condensed form for linkage
    condensed = squareform(angles_matrix, checks=False)
    
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Analyze clusters
    cluster_info = {}
    for c in range(1, n_clusters + 1):
        members = [symbols[i].symbol for i in range(len(symbols)) if clusters[i] == c]
        
        # Get roles of cluster members
        role_counts = {"from": 0, "through": 0, "to": 0, "observation": 0, "result": 0}
        for sym in symbols:
            if sym.symbol in members:
                for role, count in sym.roles.items():
                    role_counts[role] = role_counts.get(role, 0) + count
        
        # Find dominant role
        dominant_role = max(role_counts.items(), key=lambda x: x[1])[0] if role_counts else "none"
        
        cluster_info[f"cluster_{c}"] = {
            "size": len(members),
            "members": members,
            "dominant_role": dominant_role,
            "role_distribution": role_counts,
        }
    
    # Check if cluster sizes follow Fibonacci-like pattern
    sizes = sorted([c["size"] for c in cluster_info.values()], reverse=True)
    size_ratios = [sizes[i]/sizes[i+1] if sizes[i+1] > 0 else 0 for i in range(len(sizes)-1)]
    
    return {
        "n_clusters": n_clusters,
        "clusters": cluster_info,
        "cluster_sizes": sizes,
        "size_ratios": size_ratios,
        "mean_size_ratio": float(np.mean([r for r in size_ratios if r > 0])),
    }


def analyze_role_geometry(
    symbols: List[SymbolData],
    embeddings: np.ndarray
) -> Dict:
    """
    Analyze how symbols with different S64 roles are distributed.
    
    In S64, each symbol can play roles: from, through, to, observation, result.
    Do these form geometric patterns?
    """
    role_embeddings = {"from": [], "through": [], "to": [], "observation": [], "result": []}
    role_symbols = {"from": [], "through": [], "to": [], "observation": [], "result": []}
    
    for i, sym in enumerate(symbols):
        for role, count in sym.roles.items():
            if count > 0 and role in role_embeddings:
                role_embeddings[role].append(embeddings[i])
                role_symbols[role].append(sym.symbol)
    
    results = {}
    
    for role, embs in role_embeddings.items():
        if len(embs) < 2:
            continue
        
        embs_arr = np.array(embs)
        
        # Compute centroid
        centroid = np.mean(embs_arr, axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        
        # Compute angles from centroid
        norms = np.linalg.norm(embs_arr, axis=1, keepdims=True)
        normalized = embs_arr / np.where(norms > 0, norms, 1)
        cos_to_centroid = normalized @ centroid_norm
        angles_to_centroid = np.degrees(np.arccos(np.clip(cos_to_centroid, -1, 1)))
        
        results[role] = {
            "count": len(embs),
            "symbols": role_symbols[role][:10],  # First 10 for brevity
            "mean_angle_to_centroid": float(np.mean(angles_to_centroid)),
            "std_angle_to_centroid": float(np.std(angles_to_centroid)),
        }
    
    # Compute angles between role centroids
    role_centroids = {}
    for role, embs in role_embeddings.items():
        if len(embs) > 0:
            centroid = np.mean(np.array(embs), axis=0)
            role_centroids[role] = centroid / np.linalg.norm(centroid)
    
    centroid_angles = {}
    roles = list(role_centroids.keys())
    for i in range(len(roles)):
        for j in range(i+1, len(roles)):
            cos_sim = np.dot(role_centroids[roles[i]], role_centroids[roles[j]])
            angle = math.degrees(math.acos(np.clip(cos_sim, -1, 1)))
            centroid_angles[f"{roles[i]}-{roles[j]}"] = float(angle)
    
    results["centroid_angles"] = centroid_angles
    
    return results


def compute_dimensionality_metrics(embeddings: np.ndarray) -> Dict:
    """
    Analyze the effective dimensionality of the symbol space.
    
    Even though embeddings are 768D, the symbols may live in a lower-dimensional subspace.
    """
    # Center the data
    centered = embeddings - np.mean(embeddings, axis=0)
    
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Explained variance ratio
    explained_var = (S ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(explained_var)
    
    # Effective dimensionality: number of components for 90%, 95%, 99% variance
    dims_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
    dims_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
    dims_99 = int(np.argmax(cumulative_var >= 0.99) + 1)
    
    # Participation ratio (alternative measure)
    participation_ratio = (np.sum(S ** 2) ** 2) / np.sum(S ** 4)
    
    # Check if top singular values follow Fibonacci/power law
    if len(S) >= 10:
        top_10_ratios = [S[i]/S[i+1] if S[i+1] > 0 else 0 for i in range(9)]
    else:
        top_10_ratios = []
    
    return {
        "embedding_dim": embeddings.shape[1],
        "n_symbols": embeddings.shape[0],
        "dims_for_90pct_var": dims_90,
        "dims_for_95pct_var": dims_95,
        "dims_for_99pct_var": dims_99,
        "participation_ratio": float(participation_ratio),
        "top_10_explained_var": [float(v) for v in explained_var[:10]],
        "top_10_singular_value_ratios": [float(r) for r in top_10_ratios],
    }


def find_symbol_neighborhoods(
    symbols: List[SymbolData],
    angles_matrix: np.ndarray,
    k: int = 5
) -> Dict:
    """
    Find the k nearest neighbors for each symbol.
    
    This shows the semantic neighborhood structure.
    """
    neighborhoods = {}
    
    for i, sym in enumerate(symbols):
        # Get angles to all other symbols
        angles = angles_matrix[i].copy()
        angles[i] = float('inf')  # Exclude self
        
        # Find k smallest angles
        nearest_indices = np.argsort(angles)[:k]
        
        neighborhoods[sym.symbol] = [
            {"symbol": symbols[j].symbol, "angle": float(angles[j])}
            for j in nearest_indices
        ]
    
    return neighborhoods


def plot_analysis(
    angles_flat: np.ndarray,
    fibonacci_results: Dict,
    cluster_results: Dict,
    random_baseline: Optional[Dict],
    output_dir: Path
) -> None:
    """Generate visualizations of the analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Angular distance histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n, bins, patches = ax.hist(angles_flat, bins=60, edgecolor='black', alpha=0.7)
    
    # Optional: mark random-baseline mean (more actionable than golden-angle markers for this dataset)
    if random_baseline and "random_baseline" in random_baseline:
        rb_mean = random_baseline["random_baseline"]["pairwise_angle_mean_deg"]["mean"]
        ax.axvline(x=rb_mean, color='purple', linestyle='--', linewidth=2,
                   label=f'Random baseline mean ({rb_mean:.1f}deg)')
    
    ax.set_xlabel('Angular Distance (degrees)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Angular Distances Between S64 Symbols', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'angular_distribution.png', dpi=150)
    plt.close()
    print(f"[PLOT] Saved: angular_distribution.png")
    
    # 2. Cluster size distribution
    sizes = cluster_results["cluster_sizes"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(1, len(sizes)+1), sizes, color='steelblue', edgecolor='black')
    ax.set_xlabel('Cluster (by size)', fontsize=12)
    ax.set_ylabel('Number of Symbols', fontsize=12)
    ax.set_title('S64 Symbol Cluster Sizes', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate with cluster inequality (gini) if available
    mean_ratio = cluster_results.get("mean_size_ratio", float("nan"))
    gini = cluster_results.get("gini", None)
    gini_str = f"\nGini: {gini:.3f}" if isinstance(gini, (int, float)) else ""
    ax.annotate(f'Mean size ratio: {mean_ratio:.3f}{gini_str}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_sizes.png', dpi=150)
    plt.close()
    print(f"[PLOT] Saved: cluster_sizes.png")
    
    # 3. 2D projection of symbols
    # We'll use the first 2 principal components
    # (This was computed earlier; for now, skip if embeddings not passed)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze S64 Symbol Embedding Geometry")
    parser.add_argument("--backend", nargs="+", default=["s128"], 
                       help="Embedding backend(s) to analyze (can specify multiple)")
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--baseline-samples", type=int, default=DEFAULT_RANDOM_BASELINE_SAMPLES,
                        help="Number of random hypersphere baseline samples to draw (per backend)")
    parser.add_argument("--role-permutations", type=int, default=DEFAULT_ROLE_PERMUTATION_SAMPLES,
                        help="Number of permutations for role separation test (per backend)")
    parser.add_argument("--save-role-null-samples", action="store_true",
                        help="Save per-permutation null angle samples in output JSON (enables exact two-sided p-values downstream).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for baselines/permutations")
    parser.add_argument("--modes", nargs="+", default=["raw", "centered"],
                        choices=["raw", "centered", "whitened"],
                        help="Embedding preprocessing modes to analyze. Default: raw + centered.")
    parser.add_argument("--legacy-golden", action="store_true",
                        help="Include legacy golden-ratio heuristics in output/plots (not recommended)")
    args = parser.parse_args()
    
    # Ensure backend is a list
    if isinstance(args.backend, str):
        backends = [args.backend]
    else:
        backends = args.backend
    
    # Process each backend
    for backend in backends:
        # Output directory
        if args.output:
            output_dir = Path(args.output) / backend
        else:
            output_dir = validation_dir / "outputs" / "symbol_geometry" / backend
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("S64 SYMBOL EMBEDDING GEOMETRY ANALYSIS")
        print("=" * 70)
        print(f"Backend: {backend}")
        print(f"Output: {output_dir}")
        print()
        
        # Fetch symbols
        print("Fetching symbol embeddings from database...")
        symbols = fetch_symbols(backend)
        print(f"Loaded {len(symbols)} symbols")
        
        if len(symbols) == 0:
            print(f"[ERROR] No symbols found for backend {backend}. Skipping...")
            continue
        
        # Convert to numpy array (raw, not normalized)
        embeddings_raw = np.array([s.embedding for s in symbols], dtype=np.float64)
        print(f"Embedding dimension: {embeddings_raw.shape[1]}")
        print()

        mode_results: Dict[str, Dict] = {}

        for mode in args.modes:
            print("-" * 70)
            print(f"MODE: {mode}")
            print("-" * 70)

            embeddings = preprocess_embeddings(embeddings_raw, mode=mode)

            # 0. Random baseline + concentration
            print("Comparing to random hypersphere baseline...")
            baseline = compare_to_random_baseline(
                embeddings,
                n_samples=int(args.baseline_samples),
                seed=int(args.seed),
            )
            obs = baseline["observed"]
            pct = baseline["random_baseline"]["percentile_of_observed"]
            print(f"  Mean pairwise angle (obs): {obs['pairwise_angle_mean_deg']:.2f}deg "
                  f"(percentile vs random: {pct['pairwise_angle_mean_deg']:.1f}%)")
            print(f"  Mean resultant length R̄:  {obs['mean_resultant_length_rbar']:.4f} "
                  f"(percentile vs random: {pct['mean_resultant_length_rbar']:.1f}%)")
            if obs.get("kappa_estimate") is not None:
                print(f"  κ estimate (rough):        {obs['kappa_estimate']:.2f}")
            print()

            # 1. Angular distances
            print("Computing angular distances...")
            angles_matrix, angles_flat = compute_angular_distances(embeddings)
            angular_stats = analyze_angular_distribution(angles_flat)
        
            print(f"  Min angle:    {angular_stats['min_angle']:.2f}deg")
            print(f"  Max angle:    {angular_stats['max_angle']:.2f}deg")
            print(f"  Mean angle:   {angular_stats['mean_angle']:.2f}deg")
            print(f"  Median angle: {angular_stats['median_angle']:.2f}deg")
            print(f"  Std dev:      {angular_stats['std_angle']:.2f}deg")
            print()
        
            # 2. Legacy golden-ratio heuristics (disabled by default)
            fibonacci_results = {}
            if args.legacy_golden:
                print("Checking legacy Fibonacci/Golden Ratio heuristics (optional)...")
                fibonacci_results = check_fibonacci_patterns(angles_flat)

            ga = fibonacci_results.get("golden_angle", {})
            hg = fibonacci_results.get("half_golden", {})
            if ga:
                print(f"  Golden angle ({ga['target']:.1f}deg +/- {ga['tolerance']:.0f}): "
                      f"{ga['count_near']} pairs ({ga['percentage']:.1f}%)")
            if hg:
                print(f"  Half golden ({hg['target']:.1f}deg +/- 5deg): "
                      f"{hg['count_near']} pairs ({hg['percentage']:.1f}%)")

            peaks = fibonacci_results.get("distribution_peaks", [])
            if peaks:
                print("  Top distribution peaks (angle, count):")
                for angle, count in peaks[:5]:
                    print(f"    {angle:.0f}deg: {count} pairs")
                print()
        
            # 3. Semantic clusters
            print("Finding semantic clusters...")
            cluster_results = find_semantic_clusters(symbols, angles_matrix, n_clusters=12)
            cluster_results["gini"] = gini_coefficient(cluster_results.get("cluster_sizes", []))
        
            print(f"  Number of clusters: {cluster_results['n_clusters']}")
            print(f"  Cluster sizes: {cluster_results['cluster_sizes']}")
            print(f"  Cluster size Gini: {cluster_results['gini']:.3f}")
            print()

            # 3b. Cluster sweep (how stable is clumpiness across k?)
            print("Cluster sweep (clumpiness across k)...")
            sweep = cluster_sweep(symbols, angles_matrix, cluster_counts=[6, 8, 12, 16, 24])
            for k, info in sweep.items():
                print(f"  k={k}: max_cluster={info['max_cluster_fraction']:.2f}, "
                      f"gini={info['gini']:.3f}, eff_clusters={info['inv_simpson_effective_clusters']:.2f}")
            print()
        
            # 4. Role geometry (plus permutation test)
            print("Analyzing role geometry...")
            role_results = analyze_role_geometry(symbols, embeddings)
            role_angles_unique = role_centroid_angles_unique(symbols, embeddings)
            role_perm = permutation_test_role_separation(
                symbols,
                embeddings,
                n_permutations=int(args.role_permutations),
                seed=int(args.seed),
                save_null_distribution_values=bool(args.save_role_null_samples),
            )

            print("  Role centroid angles (unique membership):")
            for pair, angle in role_angles_unique.items():
                print(f"    {pair}: {angle:.2f}deg")
            print("  Role separation permutation test (unusually close p-values):")
            for pair, p in (role_perm.get("pvalue_unusually_close") or {}).items():
                if p is None:
                    continue
                print(f"    {pair}: p={p:.4f} (smaller = more unusually close)")
            print("  Role separation permutation test (unusually far p-values):")
            for pair, p in (role_perm.get("pvalue_unusually_far") or {}).items():
                if p is None:
                    continue
                print(f"    {pair}: p={p:.4f} (smaller = more unusually far)")
            print()
        
            # 5. Dimensionality
            print("Computing dimensionality metrics...")
            dim_results = compute_dimensionality_metrics(embeddings)
        
            print(f"  Embedding dimension: {dim_results['embedding_dim']}")
            print(f"  Effective dims (90% var): {dim_results['dims_for_90pct_var']}")
            print(f"  Effective dims (95% var): {dim_results['dims_for_95pct_var']}")
            print(f"  Participation ratio: {dim_results['participation_ratio']:.1f}")
            print()
        
            # 6. Neighborhoods
            print("Finding symbol neighborhoods...")
            neighborhoods = find_symbol_neighborhoods(symbols, angles_matrix, k=5)
        
            # Show a few examples
            example_symbols = ["hope", "fear", "love", "understanding", "question"]
            for sym in example_symbols:
                if sym in neighborhoods:
                    neighbors = neighborhoods[sym]
                    neighbor_str = ", ".join([f"{n['symbol']} ({n['angle']:.1f}deg)" for n in neighbors])
                    print(f"  {sym}: {neighbor_str}")
            print()

            mode_results[mode] = {
                "angular_distribution": angular_stats,
                "fibonacci_patterns": fibonacci_results,  # legacy, only present if --legacy-golden
                "clusters": cluster_results,
                "cluster_sweep": sweep,
                "role_geometry": role_results,
                "role_geometry_unique": {"centroid_angles": role_angles_unique},
                "role_permutation_test": role_perm,
                "dimensionality": dim_results,
                "neighborhoods": neighborhoods,
                "random_baseline": baseline,
            }
        
        # Save results (all modes in one file)
        results = {
            "backend": backend,
            "n_symbols": len(symbols),
            "embedding_dim": int(embeddings_raw.shape[1]),
            "modes": list(args.modes),
            "by_mode": mode_results,
            "constants": {
                "golden_ratio": GOLDEN_RATIO,
                "golden_angle_deg": GOLDEN_ANGLE_DEG,
            }
        }
        
        output_file = output_dir / f"symbol_geometry_{backend}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
        
        # Generate plots (raw-mode only for now to keep output simple)
        if args.plot and "raw" in mode_results:
            print("\nGenerating plots (raw mode)...")
            raw = mode_results["raw"]
            plot_analysis(angles_flat, raw.get("fibonacci_patterns", {}), raw["clusters"], raw["random_baseline"], output_dir)
        
        print("\n" + "=" * 70)
        print(f"ANALYSIS COMPLETE for {backend}")
        print("=" * 70)
        
        # Summary interpretation
        print("\nINTERPRETATION (compare RAW vs CENTERED):")
        print("-" * 50)
        if "raw" in mode_results and "centered" in mode_results:
            raw_rb = mode_results["raw"]["random_baseline"]
            cen_rb = mode_results["centered"]["random_baseline"]
            print(f"- RAW   mean angle: {raw_rb['observed']['pairwise_angle_mean_deg']:.2f}deg, "
                  f"R̄={raw_rb['observed']['mean_resultant_length_rbar']:.4f}")
            print(f"- CENT  mean angle: {cen_rb['observed']['pairwise_angle_mean_deg']:.2f}deg, "
                  f"R̄={cen_rb['observed']['mean_resultant_length_rbar']:.4f}")
            print("  If CENTERED jumps closer to ~90deg and R̄ drops near ~0, the 'cone' is mostly embedding anisotropy.")
            print("  If CENTERED stays low-angle / high-R̄, the cone is an intrinsic property of your symbol set.")
        else:
            print("- Run with --modes raw centered to interpret anisotropy vs intrinsic structure.")
        
        print()  # Blank line between backends
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

