#!/usr/bin/env python3
"""
export_manifold_dynamics_csv.py

Exports per-turn manifold dynamics for Tableau animation.
Computes TRUE per-turn SGI (not conversation-level aggregates).

Output schema:
- conversation_id, conversation_label, backend, mode
- turn_index, speaker
- SGI (per-turn grounding ratio)
- Velocity (step angle from previous turn)
- d_SGI, d_Velocity (deltas for phase detection)
- phase (G=Grounding, D=Discovery)
- top_symbol, symbol_confidence
- manifold_distance (how far from ideal SGI=1, Velocity=mean)

Usage:
    python export_manifold_dynamics_csv.py \
        --pack path/to/rosetta_conversations_pack.json \
        --output manifold_dynamics.csv
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional
import numpy as np


def angular_distance(v1: list, v2: list) -> float:
    """Compute angular distance in degrees between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Normalize
    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2) + 1e-10)
    # Clamp dot product to [-1, 1]
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_sgi(response_vec: list, query_vec: list, context_vec: list) -> float:
    """
    Compute SGI = θ(r, q) / θ(r, c)
    
    Where:
    - r = response embedding
    - q = query (user message)
    - c = context (mean of prior messages)
    
    Returns:
    - SGI < 1: response closer to query (query-focused)
    - SGI > 1: response closer to context (context-anchored)
    - SGI ≈ 1: balanced
    """
    theta_rq = angular_distance(response_vec, query_vec)
    theta_rc = angular_distance(response_vec, context_vec)
    
    # Avoid division by zero
    if theta_rc < 0.1:
        return 1.0  # If response is very close to context, return balanced
    
    return theta_rq / theta_rc


def extract_baseline_id(conv_id: str) -> str:
    """Extract a short baseline ID from the conversation path."""
    # Examples:
    # "naturalistic/B10_self_discovery_AI/baseline10" -> "B10"
    # "synthetic/B1_surface_deception/baseline01" -> "B01"
    # "extra::S64_LC1_expanded_free_chat" -> "LC1"
    
    if "LC1" in conv_id:
        return "LC1"
    
    # Try to find B## pattern
    import re
    match = re.search(r'[Bb](\d+)', conv_id)
    if match:
        num = int(match.group(1))
        return f"B{num:02d}"
    
    # Fallback: last part of path
    parts = conv_id.replace("::", "/").split("/")
    return parts[-1][:6] if parts else conv_id[:6]


def main():
    parser = argparse.ArgumentParser(description="Export manifold dynamics CSV")
    parser.add_argument("--pack", required=True, help="Path to rosetta_conversations_pack.json")
    parser.add_argument("--output", default="manifold_dynamics.csv", help="Output CSV path")
    parser.add_argument("--mode", default="centered", choices=["raw", "centered"], 
                        help="Which embedding mode to use (default: centered)")
    parser.add_argument("--sgi-clip", type=float, default=5.0,
                        help="Clip SGI values for visualization (default: 5.0)")
    args = parser.parse_args()
    
    pack_path = Path(args.pack)
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack not found: {pack_path}")
    
    print(f"Loading pack from {pack_path}...")
    with open(pack_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    
    rows = []
    conversations = pack.get("conversations", {})
    
    # Global stats for phase labeling
    all_sgi = []
    all_vel = []
    
    # First pass: compute all SGI and velocity values
    print("First pass: computing per-turn metrics...")
    
    for conv_id, conv_data in conversations.items():
        label = conv_data.get("label", conv_id)
        traces = conv_data.get("traces", {})
        
        for backend, backend_data in traces.items():
            mode_data = backend_data.get(args.mode, {})
            coords = mode_data.get("coords_3d", [])
            nearest_symbols = mode_data.get("nearest_symbols", [])
            
            if not coords or len(coords) < 2:
                continue
            
            n_turns = len(coords)
            
            # Compute per-turn SGI and velocity
            turn_data = []
            context_vectors = []  # Rolling context
            
            for i in range(n_turns):
                current_vec = coords[i]
                
                # Velocity: angular distance from previous turn
                if i == 0:
                    velocity = 0.0
                else:
                    velocity = angular_distance(current_vec, coords[i-1])
                
                # SGI: only meaningful for assistant turns (odd indices typically)
                # We need query (previous user turn) and context (mean of all prior)
                if i == 0:
                    sgi = 1.0  # First turn has no context
                elif i == 1:
                    # Second turn: query is first turn, no prior context
                    sgi = 1.0
                else:
                    query_vec = coords[i-1]  # Previous turn is the query
                    if len(context_vectors) > 0:
                        context_vec = np.mean(context_vectors, axis=0).tolist()
                        sgi = compute_sgi(current_vec, query_vec, context_vec)
                    else:
                        sgi = 1.0
                
                # Add current to context for next iteration
                context_vectors.append(current_vec)
                
                # Get top symbol
                if i < len(nearest_symbols) and len(nearest_symbols[i]) > 0:
                    top_sym = nearest_symbols[i][0]
                    top_symbol = top_sym.get("symbol", "unknown")
                    symbol_confidence = top_sym.get("score", 0.0)
                else:
                    top_symbol = "unknown"
                    symbol_confidence = 0.0
                
                # Determine speaker (alternating, starting with AI or User)
                speaker = "AI" if i % 2 == 0 else "User"
                
                # Clipped SGI for visualization
                sgi_clipped = min(sgi, args.sgi_clip)
                
                turn_data.append({
                    "conversation_id": conv_id,
                    "conversation_label": label,
                    "baseline_id": extract_baseline_id(conv_id),
                    "backend": backend,
                    "mode": args.mode,
                    "turn_index": i,
                    "speaker": speaker,
                    "SGI": sgi,
                    "SGI_clipped": sgi_clipped,
                    "Velocity": velocity,
                    "top_symbol": top_symbol,
                    "symbol_confidence": symbol_confidence,
                })
                
                all_sgi.append(sgi)
                all_vel.append(velocity)
            
            rows.extend(turn_data)
    
    if not rows:
        print("No data extracted. Check pack structure.")
        return 1
    
    # Compute global means for phase labeling
    mean_sgi = np.mean(all_sgi)
    mean_vel = np.mean([v for v in all_vel if v > 0])  # Exclude zero velocities
    std_sgi = np.std(all_sgi)
    std_vel = np.std([v for v in all_vel if v > 0])
    
    print(f"Global stats: SGI={mean_sgi:.3f}±{std_sgi:.3f}, Velocity={mean_vel:.2f}°±{std_vel:.2f}°")
    
    # Second pass: add deltas, phase, and manifold distance
    print("Second pass: computing deltas and phase labels...")
    
    # Group by conversation/backend for delta computation
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row["conversation_id"], row["backend"])
        grouped[key].append((i, row))
    
    for key, turn_list in grouped.items():
        turn_list.sort(key=lambda x: x[1]["turn_index"])
        
        for j, (idx, row) in enumerate(turn_list):
            # Compute deltas
            if j == 0:
                d_sgi = 0.0
                d_vel = 0.0
            else:
                prev_row = turn_list[j-1][1]
                d_sgi = row["SGI"] - prev_row["SGI"]
                d_vel = row["Velocity"] - prev_row["Velocity"]
            
            rows[idx]["d_SGI"] = d_sgi
            rows[idx]["d_Velocity"] = d_vel
            
            # Phase labeling based on your formalization:
            # Grounding (φ_G): SGI > mean AND Velocity < mean
            # Discovery (φ_D): SGI < mean AND Velocity > mean
            # Transition: neither
            sgi = row["SGI"]
            vel = row["Velocity"]
            
            if sgi > mean_sgi and vel < mean_vel:
                phase = "G"  # Grounding
            elif sgi < mean_sgi and vel > mean_vel:
                phase = "D"  # Discovery
            else:
                phase = "T"  # Transition
            
            rows[idx]["phase"] = phase
            
            # Manifold distance: how far from ideal (SGI=1, Velocity=mean)
            # Using normalized Euclidean distance
            sgi_deviation = (sgi - 1.0) / (std_sgi + 0.01)
            vel_deviation = (vel - mean_vel) / (std_vel + 0.01)
            manifold_distance = math.sqrt(sgi_deviation**2 + vel_deviation**2)
            
            rows[idx]["manifold_distance"] = manifold_distance
            
            # On manifold: close to ideal
            rows[idx]["on_manifold"] = 1 if manifold_distance < 1.5 else 0
    
    # Write CSV
    output_path = Path(args.output)
    print(f"Writing {len(rows)} rows to {output_path}...")
    
    # Define column order
    columns = [
        "conversation_id", "conversation_label", "baseline_id", "backend", "mode",
        "turn_index", "speaker",
        "SGI", "SGI_clipped", "Velocity", "d_SGI", "d_Velocity",
        "phase", "manifold_distance", "on_manifold",
        "top_symbol", "symbol_confidence"
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = [str(row.get(col, "")) for col in columns]
            f.write(",".join(values) + "\n")
    
    print(f"Done! Exported {len(rows)} turns across {len(conversations)} conversations.")
    
    # Summary stats
    phases = [r["phase"] for r in rows]
    print(f"\nPhase distribution:")
    print(f"  Grounding (G): {phases.count('G')} ({100*phases.count('G')/len(phases):.1f}%)")
    print(f"  Discovery (D): {phases.count('D')} ({100*phases.count('D')/len(phases):.1f}%)")
    print(f"  Transition (T): {phases.count('T')} ({100*phases.count('T')/len(phases):.1f}%)")
    
    on_manifold = sum(r["on_manifold"] for r in rows)
    print(f"\nManifold membership: {on_manifold}/{len(rows)} ({100*on_manifold/len(rows):.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())

