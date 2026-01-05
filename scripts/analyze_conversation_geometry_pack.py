"""
Analyze a `rosetta_conversations_pack.json` produced by `export_conversation_geometry_pack.py`.

Goal
----
Turn the pack into a small set of quantitative artifacts you can inspect slowly:
  - per conversation × backend × mode: dynamics + symbol volatility metrics
  - per conversation × mode: cross-model agreement metrics (nearest-symbol overlap)

Outputs (written under <pack_dir>/analysis_conversation_pack/<run_id>/)
-------------------------------------------------------------------
  - run_meta.json
  - traces_metrics.json
  - traces_metrics.csv
  - cross_model_agreement.json

Run (from MirrorMind root):
  python _reports/Papers/paper02/validation/scripts/analyze_conversation_geometry_pack.py ^
    --pack _reports/Papers/paper02/validation/outputs/symbol_geometry/sidecar_sweeps/sweep_YYYYMMDD_HHMMSS/conversation_traces/rosetta_conversations_pack.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set([x for x in a if x])
    sb = set([x for x in b if x])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return float(len(inter)) / float(len(union)) if union else 0.0


def _entropy_from_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / float(total)
        ent -= p * math.log(p + 1e-12, 2)
    return float(ent)


def _summarize_sequence(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    m = sum(xs) / len(xs)
    if len(xs) >= 2:
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return {"mean": float(m), "std": float(s), "min": float(min(xs)), "max": float(max(xs))}


def _pairwise(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = math.sqrt(dx2 * dy2)
    if den <= 0:
        return None
    return float(num / den)


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    Linear regression slope y ~ a + b x, returns b. Useful to interpret directionality.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        num += dx * (y - my)
        den += dx * dx
    if den <= 0:
        return None
    return float(num / den)


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _extract_topk_symbols_per_turn(nearest_symbols: Any, *, k: int) -> List[List[str]]:
    """
    nearest_symbols is expected to be:
      [ [ {symbol, score}, ... ], [ ... ], ... ]
    returns: [ [sym1, sym2, ...], ... ] (length = n_turns)
    """
    out: List[List[str]] = []
    if not isinstance(nearest_symbols, list):
        return out
    for row in nearest_symbols:
        if not isinstance(row, list):
            out.append([])
            continue
        syms: List[str] = []
        for item in row[:k]:
            if isinstance(item, dict) and isinstance(item.get("symbol"), str):
                syms.append(item["symbol"])
        out.append(syms)
    return out


def _extract_top1_score_per_turn(nearest_symbols: Any) -> List[float]:
    scores: List[float] = []
    if not isinstance(nearest_symbols, list):
        return scores
    for row in nearest_symbols:
        if not isinstance(row, list) or not row:
            scores.append(0.0)
            continue
        top = row[0]
        if isinstance(top, dict):
            scores.append(_as_float(top.get("score"), 0.0))
        else:
            scores.append(0.0)
    return scores


def _top1_symbol_per_turn(nearest_symbols: Any) -> List[str]:
    syms: List[str] = []
    if not isinstance(nearest_symbols, list):
        return syms
    for row in nearest_symbols:
        if not isinstance(row, list) or not row:
            syms.append("")
            continue
        top = row[0]
        if isinstance(top, dict) and isinstance(top.get("symbol"), str):
            syms.append(top["symbol"])
        else:
            syms.append("")
    return syms


def _resolve_path(path_str: str, mirror_mind_root: Path) -> Path:
    """
    Resolve a path string to a Path object, trying multiple strategies:
    1. As-is (if absolute or exists relative to CWD)
    2. Relative to MirrorMind root
    3. Strip "MirrorMind\" prefix if present
    """
    p = Path(path_str)
    
    # If absolute, use as-is
    if p.is_absolute():
        return p
    
    # Try as-is relative to CWD
    if p.exists():
        return p.resolve()
    
    # Try relative to MirrorMind root
    root_path = (mirror_mind_root / path_str).resolve()
    if root_path.exists():
        return root_path
    
    # Try stripping "MirrorMind\" or "MirrorMind/" prefix
    path_str_clean = path_str.replace("MirrorMind\\", "").replace("MirrorMind/", "")
    if path_str_clean != path_str:
        p_clean = Path(path_str_clean)
        if p_clean.exists():
            return p_clean.resolve()
        root_path_clean = (mirror_mind_root / path_str_clean).resolve()
        if root_path_clean.exists():
            return root_path_clean
    
    # Return the best guess (will fail with a clear error)
    return root_path if root_path.exists() else p.resolve()


def main() -> int:
    # Detect MirrorMind root
    script_dir = Path(__file__).resolve().parent
    validation_dir = script_dir.parent
    paper02_dir = validation_dir.parent
    reports_dir = paper02_dir.parent
    mirror_mind_root = reports_dir.parent
    
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, help="Path to rosetta_conversations_pack.json")
    p.add_argument("--k", type=int, default=8, help="Top-k symbols per turn to use for overlap/volatility metrics.")
    p.add_argument("--out-dir", default=None, help="Optional output directory (default: <pack_dir>/analysis_conversation_pack/<run_id>).")
    p.add_argument("--max-conversations", type=int, default=10_000)
    args = p.parse_args()

    pack_path = _resolve_path(args.pack, mirror_mind_root)
    
    if not pack_path.exists():
        raise FileNotFoundError(
            f"rosetta_conversations_pack.json not found.\n"
            f"- Provided: {args.pack}\n"
            f"- CWD: {Path.cwd()}\n"
            f"- MirrorMind root: {mirror_mind_root}\n"
            f"- Resolved to: {pack_path}\n"
            f"\n"
            f"Tip: Use a path relative to MirrorMind root (e.g., _reports/.../rosetta_conversations_pack.json)"
        )
    
    pack = json.loads(pack_path.read_text(encoding="utf-8"))

    index = pack.get("index") or {}
    conversations_index = index.get("conversations") or []
    backends = list(index.get("backends") or [])
    modes = list(index.get("modes") or [])

    conversations = pack.get("conversations") or {}
    k = max(1, int(args.k))

    run_id = _now_id()
    default_out = pack_path.parent / "analysis_conversation_pack" / f"run_{run_id}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pack_path": str(pack_path),
        "k": k,
        "n_conversations_index": len(conversations_index),
        "n_conversations_payload": len(conversations),
        "backends": backends,
        "modes": modes,
        "source": pack.get("source") or {},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # ---- Per trace metrics ----
    traces_rows: List[Dict[str, Any]] = []

    # ---- Cross-model agreement (per conversation x mode) ----
    # For each (conversation, mode), compute pairwise backend agreement based on per-turn top-k symbol overlap.
    cross_model: Dict[str, Any] = {"by_conversation": {}}

    conv_ids = [c.get("id") for c in conversations_index if isinstance(c, dict) and isinstance(c.get("id"), str)]
    # Fallback if index missing
    if not conv_ids:
        conv_ids = sorted([cid for cid in conversations.keys() if isinstance(cid, str)])

    conv_ids = conv_ids[: int(args.max_conversations)]

    for conv_id in conv_ids:
        conv = conversations.get(conv_id) or {}
        turns = conv.get("turns") or []
        traces = conv.get("traces") or {}
        n_turns = len(turns) if isinstance(turns, list) else 0

        # cross-model container
        cross_model["by_conversation"].setdefault(conv_id, {"label": conv.get("label"), "n_turns": n_turns, "modes": {}})

        # Per backend/mode metrics
        for backend, by_mode in traces.items():
            if not isinstance(by_mode, dict):
                continue
            for mode, trace in by_mode.items():
                if not isinstance(trace, dict):
                    continue

                nearest = trace.get("nearest_symbols")
                topk_syms = _extract_topk_symbols_per_turn(nearest, k=k)
                top1_syms = _top1_symbol_per_turn(nearest)
                top1_scores = _extract_top1_score_per_turn(nearest)

                # Volatility: Jaccard(top-k) between consecutive turns
                j_consec: List[float] = []
                for i in range(1, min(len(topk_syms), n_turns)):
                    j_consec.append(_jaccard(topk_syms[i - 1], topk_syms[i]))

                # Persistence vs first turn
                j_first: List[float] = []
                if topk_syms:
                    first = topk_syms[0]
                    for i in range(min(len(topk_syms), n_turns)):
                        j_first.append(_jaccard(first, topk_syms[i]))

                # Top-1 change rate
                top1_changes = 0
                top1_valid = 0
                for i in range(1, min(len(top1_syms), n_turns)):
                    if top1_syms[i - 1] and top1_syms[i]:
                        top1_valid += 1
                        if top1_syms[i] != top1_syms[i - 1]:
                            top1_changes += 1
                top1_change_rate = float(top1_changes) / float(top1_valid) if top1_valid else 0.0

                # Symbol counts / entropy over top-1 + top-k
                counts_top1: Dict[str, int] = {}
                for s in top1_syms[:n_turns]:
                    if not s:
                        continue
                    counts_top1[s] = counts_top1.get(s, 0) + 1

                counts_topk: Dict[str, int] = {}
                for row in topk_syms[:n_turns]:
                    for s in row:
                        if not s:
                            continue
                        counts_topk[s] = counts_topk.get(s, 0) + 1

                ent_top1 = _entropy_from_counts(counts_top1)
                ent_topk = _entropy_from_counts(counts_topk)

                # 3D dynamics are already visualized in-app; here we pull 768D stats if present.
                stats_highd = trace.get("stats_highd") or {}

                row = {
                    "conversation_id": conv_id,
                    "conversation_label": conv.get("label"),
                    "backend": backend,
                    "mode": mode,
                    "n_turns": n_turns,
                    # High-D dynamics (if present)
                    "highd_R": _as_float(stats_highd.get("mean_resultant_length"), 0.0),
                    "highd_step_angle_mean_deg": _as_float(stats_highd.get("mean_step_angle_deg"), 0.0),
                    "highd_step_angle_std_deg": _as_float(stats_highd.get("std_step_angle_deg"), 0.0),
                    "highd_step_angle_min_deg": _as_float(stats_highd.get("min_step_angle_deg"), 0.0),
                    "highd_step_angle_max_deg": _as_float(stats_highd.get("max_step_angle_deg"), 0.0),
                    # Symbol dynamics
                    "top1_score_mean": _summarize_sequence(top1_scores[:n_turns])["mean"],
                    "top1_score_std": _summarize_sequence(top1_scores[:n_turns])["std"],
                    "top1_change_rate": top1_change_rate,
                    "topk_unique_symbols": int(len(counts_topk)),
                    "top1_unique_symbols": int(len(counts_top1)),
                    "top1_entropy_bits": float(ent_top1),
                    "topk_entropy_bits": float(ent_topk),
                    "consec_topk_jaccard_mean": _summarize_sequence(j_consec)["mean"],
                    "consec_topk_jaccard_std": _summarize_sequence(j_consec)["std"],
                    "to_first_topk_jaccard_mean": _summarize_sequence(j_first)["mean"],
                    "to_first_topk_jaccard_std": _summarize_sequence(j_first)["std"],
                }
                traces_rows.append(row)

        # Cross-model agreement for each mode, between all backend pairs.
        # We compute per-turn Jaccard overlap of top-k symbol sets, then average across turns.
        for mode in modes:
            per_backend_topk: Dict[str, List[List[str]]] = {}
            for backend in backends:
                trace = ((traces.get(backend) or {}).get(mode) or {})
                if not isinstance(trace, dict):
                    continue
                topk_syms = _extract_topk_symbols_per_turn(trace.get("nearest_symbols"), k=k)
                if topk_syms:
                    per_backend_topk[backend] = topk_syms[:n_turns]

            bnames = sorted(per_backend_topk.keys())
            pairs = _pairwise(bnames)
            pair_rows: List[Dict[str, Any]] = []
            for a, b in pairs:
                A = per_backend_topk[a]
                B = per_backend_topk[b]
                nn = min(len(A), len(B), n_turns)
                per_turn = [_jaccard(A[i], B[i]) for i in range(nn)]
                s = _summarize_sequence(per_turn)
                pair_rows.append(
                    {
                        "backend_a": a,
                        "backend_b": b,
                        "n_turns_compared": nn,
                        "jaccard_mean": s["mean"],
                        "jaccard_std": s["std"],
                        "jaccard_min": s["min"],
                        "jaccard_max": s["max"],
                    }
                )

            cross_model["by_conversation"][conv_id]["modes"][mode] = {
                "k": k,
                "backends_present": bnames,
                "pairwise": pair_rows,
            }

    # Save per-trace metrics
    (out_dir / "traces_metrics.json").write_text(json.dumps(traces_rows, indent=2), encoding="utf-8")

    # CSV (flat, easy to load into spreadsheets)
    csv_path = out_dir / "traces_metrics.csv"
    if traces_rows:
        fieldnames = sorted({k for row in traces_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in traces_rows:
                w.writerow(row)
    else:
        csv_path.write_text("", encoding="utf-8")

    # Save cross-model agreement
    (out_dir / "cross_model_agreement.json").write_text(json.dumps(cross_model, indent=2), encoding="utf-8")

    # ---- Cone explanation: correlations between highd_R and other per-trace metrics ----
    # Group by backend+mode across conversations, compute Pearson correlations.
    metrics_for_corr = [
        "highd_step_angle_mean_deg",
        "highd_step_angle_std_deg",
        "top1_change_rate",
        "top1_entropy_bits",
        "topk_entropy_bits",
        "top1_score_mean",
        "top1_score_std",
        "consec_topk_jaccard_mean",
        "consec_topk_jaccard_std",
        "to_first_topk_jaccard_mean",
        "to_first_topk_jaccard_std",
        "topk_unique_symbols",
        "top1_unique_symbols",
    ]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in traces_rows:
        backend = str(row.get("backend") or "")
        mode = str(row.get("mode") or "")
        if not backend or not mode:
            continue
        grouped.setdefault((backend, mode), []).append(row)

    cone_explanations: Dict[str, Any] = {"by_backend_mode": {}}
    for (backend, mode), rows in grouped.items():
        # Prepare x values
        xs_all: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            v = _finite(r.get("highd_R"))
            if v is None:
                continue
            xs_all.append((v, r))
        xs_all.sort(key=lambda t: t[0])
        if len(xs_all) < 3:
            continue

        # Correlations
        corr_rows: List[Dict[str, Any]] = []
        for metric in metrics_for_corr:
            xs: List[float] = []
            ys: List[float] = []
            for x, rr in xs_all:
                y = _finite(rr.get(metric))
                if y is None:
                    continue
                xs.append(float(x))
                ys.append(float(y))
            r = _pearson_r(xs, ys)
            b = _slope(xs, ys)
            if r is None:
                continue
            corr_rows.append(
                {
                    "metric": metric,
                    "n": int(len(xs)),
                    "pearson_r": float(r),
                    "slope": float(b) if b is not None else None,
                    "abs_r": float(abs(r)),
                }
            )
        corr_rows.sort(key=lambda d: float(d.get("abs_r") or 0.0), reverse=True)

        # Extremes (top cone / low cone)
        def _row_preview(rr: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "conversation_id": rr.get("conversation_id"),
                "conversation_label": rr.get("conversation_label"),
                "highd_R": rr.get("highd_R"),
                "highd_step_angle_mean_deg": rr.get("highd_step_angle_mean_deg"),
                "highd_step_angle_std_deg": rr.get("highd_step_angle_std_deg"),
                "top1_change_rate": rr.get("top1_change_rate"),
                "topk_entropy_bits": rr.get("topk_entropy_bits"),
                "consec_topk_jaccard_mean": rr.get("consec_topk_jaccard_mean"),
                "to_first_topk_jaccard_mean": rr.get("to_first_topk_jaccard_mean"),
                "top1_score_mean": rr.get("top1_score_mean"),
                "topk_unique_symbols": rr.get("topk_unique_symbols"),
            }

        top_cone = [_row_preview(r) for (_, r) in xs_all[-10:][::-1]]
        low_cone = [_row_preview(r) for (_, r) in xs_all[:10]]

        cone_explanations["by_backend_mode"][f"{backend}::{mode}"] = {
            "backend": backend,
            "mode": mode,
            "n_traces": int(len(xs_all)),
            "highd_R_summary": _summarize_sequence([float(x) for (x, _) in xs_all]),
            "correlations_sorted": corr_rows,
            "top_cone_traces": top_cone,
            "low_cone_traces": low_cone,
        }

    (out_dir / "cone_explanations.json").write_text(json.dumps(cone_explanations, indent=2), encoding="utf-8")

    print(f"[SAVED] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


