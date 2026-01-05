"""
Paper 02 Figure Generation Script
==================================
Generates publication-quality figures for "The Conversational Coherence Region" paper.

Figures:
  1. Cross-model Jaccard heatmap (Symbol Neighborhood Structure)
  2. Cone-ness vs Symbol Diversity scatter
  3. Raw vs Centered comparison (box plots)
  4. Conversational Coherence Region (SGI × Velocity)
  5. [Use Tableau for animation]
  6. Center of Gravity comparison
  7. Model Stability Profiles
  8. Structured vs Unstructured comparison

Usage:
    python generate_paper_figures.py

Output:
    All figures saved to ../figures/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = sns.color_palette("husl", 13)  # 13 backends
FIG_DPI = 300
FIG_FORMAT = 'png'  # or 'pdf' for publication

# Paper-friendly settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': FIG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Backend display names (shorter for plots)
BACKEND_NAMES = {
    'bge-m3': 'BGE-M3',
    'cohere-v3': 'Cohere',
    'e5-finetuned-v6': 'E5-FT',
    'google': 'Google',
    'jina-v3': 'Jina',
    'mistral-embed': 'Mistral',
    'nomic': 'Nomic',
    'openai-3-large': 'OAI-L',
    'openai-3-small': 'OAI-S',
    'openai-ada-002': 'Ada-002',
    'qwen': 'Qwen',
    's128': 'S128',
    'voyage-large-2-instruct': 'Voyage',
}

# Baseline display names
BASELINE_NAMES = {
    'B01': 'B01 Surface Deception',
    'B02': 'B02 Implicit',
    'B03': 'B03 Oscillation',
    'B04': 'B04 Stuck',
    'B05': 'B05 Nested',
    'B06': 'B06 Explicit',
    'B07': 'B07 Failed',
    'B08': 'B08 False Complete',
    'B09': 'B09 Human-AI',
    'B10': 'B10 AI-AI Structured',
    'LC1': 'LC1 AI-AI Unstructured',
}

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load all data files needed for figure generation."""
    print("Loading data files...")
    
    # Manifold dynamics (per-turn data)
    manifold_path = DATA_DIR / "manifold" / "manifold_dynamics.csv"
    manifold_df = pd.read_csv(manifold_path)
    print(f"  manifold_dynamics.csv: {len(manifold_df):,} rows")
    
    # Traces metrics (aggregated per conversation × backend × mode)
    traces_path = DATA_DIR / "conversations" / "traces_metrics.csv"
    traces_df = pd.read_csv(traces_path)
    print(f"  traces_metrics.csv: {len(traces_df):,} rows")
    
    # Cross-model agreement
    agreement_path = DATA_DIR / "conversations" / "cross_model_agreement.json"
    with open(agreement_path) as f:
        agreement_data = json.load(f)
    print(f"  cross_model_agreement.json: loaded")
    
    # Cone explanations (correlations)
    cone_path = DATA_DIR / "conversations" / "cone_explanations.json"
    with open(cone_path) as f:
        cone_data = json.load(f)
    print(f"  cone_explanations.json: loaded")
    
    return manifold_df, traces_df, agreement_data, cone_data


# ============================================================================
# Figure 1: Cross-Model Jaccard Heatmap
# ============================================================================

def figure1_jaccard_heatmap(agreement_data):
    """
    Figure 1: Symbol Neighborhood Structure
    Cross-model Jaccard agreement matrix showing which backends agree.
    """
    print("\nGenerating Figure 1: Cross-Model Jaccard Heatmap...")
    
    # Get all backends
    backends = list(BACKEND_NAMES.keys())
    n_backends = len(backends)
    
    # Build agreement matrix (average across all conversations, centered mode)
    matrix = np.zeros((n_backends, n_backends))
    counts = np.zeros((n_backends, n_backends))
    
    for conv_id, conv_data in agreement_data.get('by_conversation', {}).items():
        centered_data = conv_data.get('modes', {}).get('centered', {})
        for pair in centered_data.get('pairwise', []):
            a = pair['backend_a']
            b = pair['backend_b']
            if a in backends and b in backends:
                i, j = backends.index(a), backends.index(b)
                matrix[i, j] += pair['jaccard_mean']
                matrix[j, i] += pair['jaccard_mean']
                counts[i, j] += 1
                counts[j, i] += 1
    
    # Average
    counts[counts == 0] = 1  # avoid division by zero
    matrix = matrix / counts
    np.fill_diagonal(matrix, 1.0)  # self-agreement = 1
    
    # Create labels
    labels = [BACKEND_NAMES.get(b, b) for b in backends]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0.1,
        vmax=0.3,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Mean Jaccard Similarity', 'shrink': 0.8}
    )
    
    ax.set_title('Cross-Model Symbol Neighborhood Agreement\n(Centered Mode, k=8 Nearest Symbols)', 
                 fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add annotation for random baseline
    ax.text(0.02, -0.12, 'Random baseline (k=8, n=180): Jaccard ≈ 0.044',
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig1_jaccard_heatmap.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 2: Cone-ness vs Symbol Diversity
# ============================================================================

def figure2_coneness_vs_diversity(traces_df):
    """
    Figure 2: Cone-ness vs Symbol Diversity
    Scatter plot showing negative correlation between highd_R and topk_unique_symbols.
    """
    print("\nGenerating Figure 2: Cone-ness vs Symbol Diversity...")
    
    # Filter to centered mode only
    df = traces_df[traces_df['mode'] == 'centered'].copy()
    
    # Create backend color mapping
    unique_backends = df['backend'].unique()
    colors = dict(zip(unique_backends, sns.color_palette("husl", len(unique_backends))))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: highd_R vs topk_unique_symbols
    ax1 = axes[0]
    for backend in unique_backends:
        subset = df[df['backend'] == backend]
        ax1.scatter(
            subset['topk_unique_symbols'],
            subset['highd_R'],
            c=[colors[backend]],
            label=BACKEND_NAMES.get(backend, backend),
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Add trend line
    x = df['topk_unique_symbols']
    y = df['highd_R']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, p(x_line), '--', color='red', alpha=0.8, linewidth=2, label='Trend')
    
    # Correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax1.text(0.95, 0.95, f'r = {corr:.2f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Unique Top-k Symbols', fontsize=11)
    ax1.set_ylabel('Cone-ness (highd_R)', fontsize=11)
    ax1.set_title('(A) Symbol Diversity vs Trajectory Concentration', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=7, ncol=2)
    
    # Panel B: highd_R vs topk_entropy_bits
    ax2 = axes[1]
    for backend in unique_backends:
        subset = df[df['backend'] == backend]
        ax2.scatter(
            subset['topk_entropy_bits'],
            subset['highd_R'],
            c=[colors[backend]],
            label=BACKEND_NAMES.get(backend, backend),
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
    
    # Add trend line
    x = df['topk_entropy_bits']
    y = df['highd_R']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_line, p(x_line), '--', color='red', alpha=0.8, linewidth=2, label='Trend')
    
    corr = np.corrcoef(x, y)[0, 1]
    ax2.text(0.95, 0.95, f'r = {corr:.2f}', transform=ax2.transAxes,
             ha='right', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Symbol Entropy (bits)', fontsize=11)
    ax2.set_ylabel('Cone-ness (highd_R)', fontsize=11)
    ax2.set_title('(B) Symbol Entropy vs Trajectory Concentration', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig2_coneness_vs_diversity.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 3: Raw vs Centered Comparison
# ============================================================================

def figure3_raw_vs_centered(traces_df):
    """
    Figure 3: Raw vs Centered Comparison
    Side-by-side highd_R distributions for raw and centered modes.
    """
    print("\nGenerating Figure 3: Raw vs Centered Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Box plots by backend
    ax1 = axes[0]
    
    # Prepare data for grouped boxplot
    backends_order = sorted(traces_df['backend'].unique())
    
    sns.boxplot(
        data=traces_df,
        x='backend',
        y='highd_R',
        hue='mode',
        palette={'centered': '#3498db', 'raw': '#e74c3c'},
        ax=ax1,
        order=backends_order
    )
    
    ax1.set_xlabel('Embedding Backend', fontsize=11)
    ax1.set_ylabel('Cone-ness (highd_R)', fontsize=11)
    ax1.set_title('(A) Cone-ness by Backend and Mode', fontsize=11, fontweight='bold')
    ax1.set_xticklabels([BACKEND_NAMES.get(b, b) for b in backends_order], rotation=45, ha='right')
    ax1.legend(title='Mode', loc='upper right')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Panel B: Distribution comparison
    ax2 = axes[1]
    
    centered = traces_df[traces_df['mode'] == 'centered']['highd_R']
    raw = traces_df[traces_df['mode'] == 'raw']['highd_R']
    
    ax2.hist(centered, bins=20, alpha=0.7, label=f'Centered (μ={centered.mean():.2f})', color='#3498db')
    ax2.hist(raw, bins=20, alpha=0.7, label=f'Raw (μ={raw.mean():.2f})', color='#e74c3c')
    
    ax2.axvline(centered.mean(), color='#2980b9', linestyle='--', linewidth=2)
    ax2.axvline(raw.mean(), color='#c0392b', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Cone-ness (highd_R)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('(B) Distribution of Cone-ness Values', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left')
    
    # Add annotation
    diff = raw.mean() - centered.mean()
    ax2.text(0.95, 0.95, f'Δ = {diff:.2f}\nRaw embeddings are\nsystematically tighter',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig3_raw_vs_centered.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 4: The Conversational Coherence Region (SGI × Velocity)
# ============================================================================

def figure4_alignment_manifold(manifold_df):
    """
    Figure 4: The Conversational Coherence Region (SGI × Velocity)
    Scatter plot of all turns in SGI × Velocity space, showing phase regions.
    
    Phases from data:
    - T = Tracking (SGI close to 1, balanced grounding)
    - G = Grounding (SGI > 1.5, context-anchored)  
    - D = Drift (SGI < 0.7, query-focused / drifting from context)
    
    THE CONVERSATIONAL COHERENCE REGION is defined as:
    - SGI between 0.5 and 2.0 (balanced grounding, not too query-focused or context-anchored)
    - Velocity below ~45° (controlled movement, not erratic jumping)
    
    Turns inside this region exhibit "stable orbit" dynamics:
    - Grounded to both query and context
    - Controlled semantic movement
    - Characteristic of structured, aligned interaction
    
    Turns outside this region exhibit:
    - Drift Zone (left): Query-focused, losing context
    - Grounding Zone (right): Over-anchored to context, ignoring query
    - High velocity (top): Erratic, unstable movement
    """
    print("\nGenerating Figure 4: Conversational Coherence Region (SGI × Velocity)...")
    
    # Filter to centered mode
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Phase colors - distinct and colorblind-friendly
    phase_colors = {
        'T': '#2ecc71',  # Green - Tracking (balanced)
        'G': '#9b59b6',  # Purple - Grounding (context-anchored)
        'D': '#e74c3c',  # Red - Drift (query-focused)
    }
    phase_labels = {
        'T': f'Tracking (n={len(df[df["phase"]=="T"]):,})',
        'G': f'Grounding (n={len(df[df["phase"]=="G"]):,})',
        'D': f'Drift (n={len(df[df["phase"]=="D"]):,})',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: All data, colored by phase
    ax1 = axes[0]
    
    # Draw the "coherence region" region first (background)
    from matplotlib.patches import Rectangle
    manifold_rect = Rectangle((0.5, 0), 1.5, 45, linewidth=2, 
                               edgecolor='#27ae60', facecolor='#27ae60', 
                               alpha=0.1, linestyle='-', zorder=0)
    ax1.add_patch(manifold_rect)
    
    # Plot in specific order so important phases are on top
    for phase in ['T', 'G', 'D']:
        subset = df[df['phase'] == phase]
        if len(subset) > 0:
            ax1.scatter(
                subset['SGI_clipped'],
                subset['Velocity'],
                c=phase_colors.get(phase, 'gray'),
                label=phase_labels.get(phase, phase),
                alpha=0.5,
                s=20,
                edgecolors='none'
            )
    
    ax1.set_xlabel('SGI (Semantic Grounding Index)', fontsize=11)
    ax1.set_ylabel('Velocity (degrees per turn)', fontsize=11)
    ax1.set_title('(A) All Turns Colored by Phase', fontsize=11, fontweight='bold')
    
    # Legend with white background box
    legend = ax1.legend(loc='upper right', fontsize=9, frameon=True, 
                        facecolor='white', edgecolor='#cccccc', framealpha=0.95)
    legend.get_frame().set_linewidth(1.0)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 105)  # Slightly higher to make room for label
    
    # Add reference line with label ABOVE the plot area
    ax1.axvline(x=1.0, color='#555555', linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.text(1.0, 103, 'SGI=1 (balanced)', fontsize=9, color='#555555', 
             ha='center', va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Add phase region annotations - lowered to avoid legend overlap
    ax1.annotate('Drift Zone', xy=(0.25, 75), fontsize=10, 
                 color='#c0392b', ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.annotate('Grounding Zone', xy=(4.0, 75), fontsize=10,
                 color='#8e44ad', ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Label the coherence region
    ax1.annotate('COHERENCE\nREGION', xy=(1.25, 22), fontsize=10,
                 color='#1e8449', ha='center', fontweight='bold', alpha=0.9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#27ae60'))
    
    # Panel B: Faceted by baseline type
    ax2 = axes[1]
    
    # Identify baseline types
    df['baseline_type'] = df['baseline_id'].apply(
        lambda x: 'Naturalistic' if x in ['B09', 'B10'] else 
                  ('Free-form' if 'LC1' in str(x) or 'LC' in str(x) else 'Synthetic')
    )
    
    type_colors = {
        'Naturalistic': '#27ae60',   # Green - B09, B10
        'Free-form': '#e74c3c',      # Red - LC1
        'Synthetic': '#3498db',      # Blue - B01-B08
    }
    
    # Plot synthetic first (background), then others on top
    for btype in ['Synthetic', 'Naturalistic', 'Free-form']:
        subset = df[df['baseline_type'] == btype]
        if len(subset) > 0:
            ax2.scatter(
                subset['SGI_clipped'],
                subset['Velocity'],
                c=type_colors[btype],
                label=f'{btype} (n={len(subset):,})',
                alpha=0.5 if btype == 'Synthetic' else 0.6,
                s=15 if btype == 'Synthetic' else 25,
                edgecolors='none'
            )
    
    ax2.set_xlabel('SGI (Semantic Grounding Index)', fontsize=11)
    ax2.set_ylabel('Velocity (degrees per turn)', fontsize=11)
    ax2.set_title('(B) By Baseline Type', fontsize=11, fontweight='bold')
    
    # Legend with white background box
    legend2 = ax2.legend(loc='upper right', fontsize=9, frameon=True,
                         facecolor='white', edgecolor='#cccccc', framealpha=0.95)
    legend2.get_frame().set_linewidth(1.0)
    
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 100)
    
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig4_alignment_manifold.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 4b: Synthetic Baselines Detail (B01-B08)
# ============================================================================

def figure4b_synthetic_baselines(manifold_df):
    """
    Figure 4b: Synthetic Baselines Detail
    Shows each B01-B08 baseline separately to reveal their distinct signatures.
    """
    print("\nGenerating Figure 4b: Synthetic Baselines Detail...")
    
    # Filter to centered mode and synthetic baselines
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Get synthetic baselines (B01-B08)
    synthetic_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08']
    df_synth = df[df['baseline_id'].isin(synthetic_ids)].copy()
    
    if len(df_synth) == 0:
        print("  Warning: No synthetic baselines found!")
        return
    
    # Color palette for 8 baselines
    colors = sns.color_palette("Set2", 8)
    baseline_colors = dict(zip(synthetic_ids, colors))
    
    # Baseline descriptions (short)
    baseline_desc = {
        'B01': 'Surface Deception',
        'B02': 'Implicit Transform',
        'B03': 'Rapid Oscillation',
        'B04': 'Stuck States',
        'B05': 'Nested Complexity',
        'B06': 'Explicit Transform',
        'B07': 'Failed Transform',
        'B08': 'False Completion',
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, bid in enumerate(synthetic_ids):
        ax = axes[idx]
        subset = df_synth[df_synth['baseline_id'] == bid]
        
        if len(subset) > 0:
            # Plot trajectory as connected line
            for backend in subset['backend'].unique()[:3]:  # Limit to 3 backends for clarity
                backend_data = subset[subset['backend'] == backend].sort_values('turn_index')
                ax.plot(
                    backend_data['SGI_clipped'],
                    backend_data['Velocity'],
                    alpha=0.3,
                    linewidth=0.5,
                    color='gray'
                )
            
            # Scatter points
            ax.scatter(
                subset['SGI_clipped'],
                subset['Velocity'],
                c=[baseline_colors[bid]],
                alpha=0.6,
                s=25,
                edgecolors='white',
                linewidth=0.3
            )
            
            # Stats
            mean_sgi = subset['SGI_clipped'].mean()
            mean_vel = subset['Velocity'].mean()
            n_turns = len(subset) // len(subset['backend'].unique())  # per backend
            
            ax.scatter([mean_sgi], [mean_vel], c='black', s=100, marker='X', 
                      edgecolors='white', linewidth=1.5, zorder=10)
        
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 100)
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{bid}: {baseline_desc.get(bid, "")}\n(~{n_turns} turns)', 
                    fontsize=9, fontweight='bold')
        
        if idx >= 4:
            ax.set_xlabel('SGI', fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel('Velocity (deg)', fontsize=9)
    
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig4b_synthetic_baselines.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 6: Center of Gravity Comparison
# ============================================================================

def figure6_center_of_gravity(manifold_df):
    """
    Figure 6: Center of Gravity Comparison
    Mean SGI and Velocity for each baseline × backend combination.
    """
    print("\nGenerating Figure 6: Center of Gravity Comparison...")
    
    # Filter to centered mode
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Aggregate by baseline and backend
    agg = df.groupby(['baseline_id', 'backend']).agg({
        'SGI': ['mean', 'std'],
        'Velocity': ['mean', 'std']
    }).reset_index()
    agg.columns = ['baseline_id', 'backend', 'SGI_mean', 'SGI_std', 'Velocity_mean', 'Velocity_std']
    
    # Identify key baselines
    key_baselines = ['B09', 'B10']
    lc1_patterns = ['LC1', 'LC', 'extra']
    
    agg['baseline_type'] = agg['baseline_id'].apply(
        lambda x: 'B09 (Human-AI)' if 'B09' in str(x) else
                  ('B10 (AI-AI Structured)' if 'B10' in str(x) else
                   ('LC1 (AI-AI Unstructured)' if any(p in str(x) for p in lc1_patterns) else 'Other'))
    )
    
    # Filter to key baselines
    key_agg = agg[agg['baseline_type'] != 'Other']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    type_colors = {
        'B09 (Human-AI)': '#27ae60',
        'B10 (AI-AI Structured)': '#3498db',
        'LC1 (AI-AI Unstructured)': '#e74c3c'
    }
    
    for btype in type_colors.keys():
        subset = key_agg[key_agg['baseline_type'] == btype]
        if len(subset) > 0:
            ax.scatter(
                subset['SGI_mean'],
                subset['Velocity_mean'],
                c=type_colors[btype],
                label=btype,
                s=100,
                alpha=0.8,
                edgecolors='white',
                linewidth=1
            )
            
            # Add error bars
            ax.errorbar(
                subset['SGI_mean'],
                subset['Velocity_mean'],
                xerr=subset['SGI_std'],
                yerr=subset['Velocity_std'],
                c=type_colors[btype],
                fmt='none',
                alpha=0.3,
                capsize=2
            )
    
    # Add labels for backends
    for _, row in key_agg.iterrows():
        ax.annotate(
            BACKEND_NAMES.get(row['backend'], row['backend']),
            (row['SGI_mean'], row['Velocity_mean']),
            fontsize=7,
            alpha=0.7,
            xytext=(3, 3),
            textcoords='offset points'
        )
    
    ax.set_xlabel('Mean SGI', fontsize=11)
    ax.set_ylabel('Mean Velocity (degrees)', fontsize=11)
    ax.set_title('Center of Gravity Analysis\n(Error bars = ±1 std)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add annotation box
    textstr = '\n'.join([
        'Key Finding:',
        '• Structured (B09, B10): Low velocity, stable SGI',
        '• Unstructured (LC1): High velocity, high variance'
    ])
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig6_center_of_gravity.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 7: Model Stability Profiles
# ============================================================================

def figure7_model_stability(manifold_df):
    """
    Figure 7: Model Stability Profiles
    Comparison of trajectory spread (std SGI, std Velocity) across backends.
    """
    print("\nGenerating Figure 7: Model Stability Profiles...")
    
    # Filter to centered mode
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Aggregate by backend
    agg = df.groupby('backend').agg({
        'SGI': ['mean', 'std'],
        'Velocity': ['mean', 'std'],
        'turn_index': 'count'
    }).reset_index()
    agg.columns = ['backend', 'SGI_mean', 'SGI_std', 'Velocity_mean', 'Velocity_std', 'n_turns']
    agg = agg.sort_values('Velocity_std')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Velocity std (lower = more stable)
    ax1 = axes[0]
    colors = ['#27ae60' if x < agg['Velocity_std'].median() else '#e74c3c' 
              for x in agg['Velocity_std']]
    
    bars = ax1.barh(
        [BACKEND_NAMES.get(b, b) for b in agg['backend']],
        agg['Velocity_std'],
        color=colors,
        alpha=0.8,
        edgecolor='white'
    )
    
    ax1.set_xlabel('Velocity Std Dev (degrees)', fontsize=11)
    ax1.set_title('(A) Trajectory Stability (lower = more stable)', fontsize=11, fontweight='bold')
    ax1.axvline(agg['Velocity_std'].median(), color='gray', linestyle='--', alpha=0.7)
    
    # Panel B: SGI std
    ax2 = axes[1]
    agg_sorted = agg.sort_values('SGI_std')
    colors = ['#27ae60' if x < agg['SGI_std'].median() else '#e74c3c' 
              for x in agg_sorted['SGI_std']]
    
    ax2.barh(
        [BACKEND_NAMES.get(b, b) for b in agg_sorted['backend']],
        agg_sorted['SGI_std'],
        color=colors,
        alpha=0.8,
        edgecolor='white'
    )
    
    ax2.set_xlabel('SGI Std Dev', fontsize=11)
    ax2.set_title('(B) Grounding Stability (lower = more stable)', fontsize=11, fontweight='bold')
    ax2.axvline(agg['SGI_std'].median(), color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig7_model_stability.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 8: Structured vs Unstructured Interactions
# ============================================================================

def figure8_structured_vs_unstructured(manifold_df):
    """
    Figure 8: Structured vs Unstructured Interactions
    Side-by-side manifold plots for B09/B10 (structured) vs LC1 (unstructured).
    """
    print("\nGenerating Figure 8: Structured vs Unstructured Comparison...")
    
    # Filter to centered mode
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Classify baselines
    df['baseline_type'] = df['baseline_id'].apply(
        lambda x: 'Structured' if any(b in str(x) for b in ['B09', 'B10']) else
                  ('Unstructured' if any(p in str(x) for p in ['LC1', 'LC', 'extra']) else 'Synthetic')
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common settings
    xlim = (0, 5)
    ylim = (0, 100)
    
    # Panel A: Structured (B09, B10)
    ax1 = axes[0]
    structured = df[df['baseline_type'] == 'Structured']
    ax1.scatter(
        structured['SGI_clipped'],
        structured['Velocity'],
        c='#27ae60',
        alpha=0.4,
        s=20,
        edgecolors='none'
    )
    ax1.set_xlabel('SGI', fontsize=11)
    ax1.set_ylabel('Velocity (deg)', fontsize=11)
    ax1.set_title(f'(A) Structured (B09, B10)\nn = {len(structured):,} turns', fontsize=11, fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Add center of gravity
    cog_x, cog_y = structured['SGI_clipped'].mean(), structured['Velocity'].mean()
    ax1.scatter([cog_x], [cog_y], c='darkgreen', s=200, marker='X', edgecolors='white', linewidth=2, zorder=10)
    ax1.annotate(f'CoG: ({cog_x:.1f}, {cog_y:.0f}°)', (cog_x, cog_y), fontsize=9, 
                 xytext=(10, 10), textcoords='offset points')
    
    # Panel B: Unstructured (LC1)
    ax2 = axes[1]
    unstructured = df[df['baseline_type'] == 'Unstructured']
    ax2.scatter(
        unstructured['SGI_clipped'],
        unstructured['Velocity'],
        c='#e74c3c',
        alpha=0.4,
        s=20,
        edgecolors='none'
    )
    ax2.set_xlabel('SGI', fontsize=11)
    ax2.set_ylabel('Velocity (deg)', fontsize=11)
    ax2.set_title(f'(B) Unstructured (LC1)\nn = {len(unstructured):,} turns', fontsize=11, fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    cog_x, cog_y = unstructured['SGI_clipped'].mean(), unstructured['Velocity'].mean()
    ax2.scatter([cog_x], [cog_y], c='darkred', s=200, marker='X', edgecolors='white', linewidth=2, zorder=10)
    ax2.annotate(f'CoG: ({cog_x:.1f}, {cog_y:.0f}°)', (cog_x, cog_y), fontsize=9,
                 xytext=(10, 10), textcoords='offset points')
    
    # Panel C: Overlay comparison
    ax3 = axes[2]
    ax3.scatter(
        structured['SGI_clipped'],
        structured['Velocity'],
        c='#27ae60',
        alpha=0.3,
        s=15,
        label='Structured',
        edgecolors='none'
    )
    ax3.scatter(
        unstructured['SGI_clipped'],
        unstructured['Velocity'],
        c='#e74c3c',
        alpha=0.3,
        s=15,
        label='Unstructured',
        edgecolors='none'
    )
    ax3.set_xlabel('SGI', fontsize=11)
    ax3.set_ylabel('Velocity (deg)', fontsize=11)
    ax3.set_title('(C) Overlay Comparison', fontsize=11, fontweight='bold')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig8_structured_vs_unstructured.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 5: Trajectory Evolution (Static version for PDF)
# ============================================================================

def figure5_trajectory_b09(manifold_df):
    """
    Figure 5: B09 Trajectory through the Conversational Coherence Region
    Static version with color gradient showing temporal evolution.
    """
    print("\nGenerating Figure 5: B09 Trajectory Evolution...")
    
    # Filter to B09, centered mode, single backend (Nomic for stability)
    df = manifold_df[
        (manifold_df['mode'] == 'centered') & 
        (manifold_df['baseline_id'] == 'B09')
    ].copy()
    
    if len(df) == 0:
        # Try alternative naming
        df = manifold_df[
            (manifold_df['mode'] == 'centered') & 
            (manifold_df['baseline_id'].str.contains('B09', na=False))
        ].copy()
    
    if len(df) == 0:
        print("  Warning: No B09 data found!")
        return
    
    # Use a stable backend
    preferred_backends = ['nomic', 'google', 's128']
    backend = None
    for b in preferred_backends:
        if b in df['backend'].values:
            backend = b
            break
    if backend is None:
        backend = df['backend'].iloc[0]
    
    df_backend = df[df['backend'] == backend].sort_values('turn_index')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the coherence region region
    from matplotlib.patches import Rectangle
    manifold_rect = Rectangle((0.5, 0), 1.5, 45, linewidth=2, 
                               edgecolor='#27ae60', facecolor='#27ae60', 
                               alpha=0.1, linestyle='-', zorder=0)
    ax.add_patch(manifold_rect)
    
    # Get trajectory data
    x = df_backend['SGI_clipped'].values
    y = df_backend['Velocity'].values
    turns = df_backend['turn_index'].values
    n_turns = len(turns)
    
    # Create color gradient (dark to bright)
    colors = plt.cm.viridis(np.linspace(0.2, 1.0, n_turns))
    
    # Draw trajectory line with gradient
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], 
                color=colors[i], linewidth=1.5, alpha=0.7, zorder=1)
    
    # Scatter points with gradient
    scatter = ax.scatter(x, y, c=turns, cmap='viridis', s=50, 
                        edgecolors='white', linewidth=0.5, zorder=2,
                        vmin=0, vmax=n_turns)
    
    # Mark start and end
    ax.scatter([x[0]], [y[0]], c='#2ecc71', s=150, marker='o', 
              edgecolors='black', linewidth=2, zorder=10, label='Start (Turn 1)')
    ax.scatter([x[-1]], [y[-1]], c='#e74c3c', s=150, marker='s', 
              edgecolors='black', linewidth=2, zorder=10, label=f'End (Turn {n_turns})')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Turn Number', fontsize=10)
    
    # Reference line
    ax.axvline(x=1.0, color='#555555', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(1.0, 97, 'SGI=1', fontsize=9, color='#555555', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Labels
    ax.set_xlabel('SGI (Semantic Grounding Index)', fontsize=11)
    ax.set_ylabel('Velocity (degrees per turn)', fontsize=11)
    ax.set_title(f'B09 Trajectory Through the Coherence Region\n'
                 f'({n_turns} turns, {BACKEND_NAMES.get(backend, backend)} backend)',
                 fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 100)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=9, frameon=True,
                      facecolor='white', edgecolor='#cccccc', framealpha=0.95)
    
    # Coherence region label
    ax.annotate('COHERENCE\nREGION', xy=(1.25, 22), fontsize=10,
                color='#1e8449', ha='center', fontweight='bold', alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#27ae60'))
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig5_trajectory_b09.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI)
    plt.close()
    print(f"  Saved: {filepath}")
    print(f"  Backend: {backend}, Turns: {n_turns}")


# ============================================================================
# Export for Tableau (Figure 5 animation - supplementary)
# ============================================================================

def export_tableau_data(manifold_df):
    """
    Export data formatted for Tableau animation (Figure 5 web version).
    """
    print("\nExporting Tableau data for Figure 5 animation...")
    
    # Filter to key baselines and centered mode
    df = manifold_df[manifold_df['mode'] == 'centered'].copy()
    
    # Select relevant columns
    cols = ['conversation_id', 'baseline_id', 'backend', 'turn_index', 'speaker',
            'SGI', 'SGI_clipped', 'Velocity', 'phase', 'top_symbol', 'symbol_confidence']
    
    tableau_df = df[cols].copy()
    
    # Add display names
    tableau_df['backend_display'] = tableau_df['backend'].map(BACKEND_NAMES)
    
    filepath = FIGURES_DIR / "fig5_tableau_animation_data.csv"
    tableau_df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    print(f"  Rows: {len(tableau_df):,}")
    print("\n  Tableau setup (for web animation):")
    print("    - Columns: turn_index (Pages), SGI_clipped (X), Velocity (Y)")
    print("    - Color: backend_display or phase")
    print("    - Path: line connecting turns")
    print("    - Filter: baseline_id = 'B09' for single trajectory")


# ============================================================================
# Figure: Role Geometry
# ============================================================================

def figure_role_geometry():
    """
    Figure: Role Geometry Permutation Tests
    
    Shows that S64 grammatical roles (from, through, to, result) exhibit
    consistent geometric relationships across all embedding backends.
    """
    print("\nGenerating Figure: Role Geometry...")
    
    # Load sweep summary which contains role permutation data
    sweep_path = DATA_DIR / "sweep" / "sweep_summary.json"
    
    if not sweep_path.exists():
        print(f"  ⚠️ Sweep summary not found: {sweep_path}")
        return
    
    with open(sweep_path) as f:
        sweep_data = json.load(f)
    
    # Extract role geometry p-values for each backend
    backends_data = []
    
    for backend_name, backend_info in sweep_data.get('backends', {}).items():
        by_mode = backend_info.get('by_mode_summary', {})
        
        for mode in ['raw', 'centered']:
            if mode not in by_mode:
                continue
            
            mode_data = by_mode[mode]
            
            # Get role permutation p-values
            p_close = mode_data.get('role_perm_p_close', {})
            p_far = mode_data.get('role_perm_p_far', {})
            
            if p_close and p_far:
                backends_data.append({
                    'backend': BACKEND_NAMES.get(backend_name, backend_name),
                    'mode': mode,
                    'from_to_p_close': p_close.get('from-to', None),
                    'through_result_p_close': p_close.get('through-result', None),
                    'from_to_p_far': p_far.get('from-to', None),
                    'through_result_p_far': p_far.get('through-result', None),
                })
    
    df = pd.DataFrame(backends_data)
    
    if df.empty:
        print("  ⚠️ No role geometry data found")
        return
    
    # Filter to raw mode (clearer signal)
    df_raw = df[df['mode'] == 'raw'].copy()
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort by backend name for consistency
    df_raw = df_raw.sort_values('backend')
    backends = df_raw['backend'].values
    x = np.arange(len(backends))
    
    # Transform p-values to -log10(p) for visualization
    # p=0.05 -> 1.3, p=0.001 -> 3, p=0 -> cap at 4 (representing p < 0.0001)
    def neg_log10_p(p):
        if p == 0 or p < 0.0001:
            return 4.0  # Cap at 4 (represents p < 0.0001)
        return -np.log10(p)
    
    # Panel A: from-to pairs (should be CLOSE, so p_close should be low)
    ax1 = axes[0]
    p_values = df_raw['from_to_p_close'].values
    log_p = [neg_log10_p(p) for p in p_values]
    
    colors = ['#27ae60' if p < 0.05 else '#95a5a6' for p in p_values]
    bars1 = ax1.bar(x, log_p, color=colors, edgecolor='white', linewidth=0.5)
    
    # Significance threshold line: -log10(0.05) ≈ 1.3
    ax1.axhline(y=1.3, color='#e74c3c', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backends, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel(r'$-\log_{10}$(p-value)', fontsize=11)
    ax1.set_title('(A) from ↔ to: Significantly CLOSER\nthan random permutation', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.legend(loc='upper right', fontsize=9)
    
    # Add p-value labels on bars
    for i, (lp, p) in enumerate(zip(log_p, p_values)):
        label = 'p<.0001' if p < 0.0001 else f'p={p:.3f}'
        ax1.annotate(label, xy=(i, lp + 0.15), ha='center', fontsize=7, rotation=90)
    
    # Panel B: through-result pairs (should be FAR, so p_far should be low)
    ax2 = axes[1]
    p_values_far = df_raw['through_result_p_far'].values
    log_p_far = [neg_log10_p(p) for p in p_values_far]
    
    colors2 = ['#3498db' if p < 0.05 else '#95a5a6' for p in p_values_far]
    bars2 = ax2.bar(x, log_p_far, color=colors2, edgecolor='white', linewidth=0.5)
    
    ax2.axhline(y=1.3, color='#e74c3c', linestyle='--', linewidth=2, label='α = 0.05')
    ax2.set_xticks(x)
    ax2.set_xticklabels(backends, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel(r'$-\log_{10}$(p-value)', fontsize=11)
    ax2.set_title('(B) through ↔ result: Significantly FARTHER\nthan random permutation', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 5)
    ax2.legend(loc='upper right', fontsize=9)
    
    # Add p-value labels on bars
    for i, (lp, p) in enumerate(zip(log_p_far, p_values_far)):
        label = 'p<.0001' if p < 0.0001 else f'p={p:.3f}'
        ax2.annotate(label, xy=(i, lp + 0.15), ha='center', fontsize=7, rotation=90)
    
    plt.tight_layout()
    
    filepath = FIGURES_DIR / f"fig_role_geometry.{FIG_FORMAT}"
    plt.savefig(filepath, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Paper 02 Figure Generation")
    print("=" * 60)
    
    # Load data
    manifold_df, traces_df, agreement_data, cone_data = load_data()
    
    # Generate figures
    figure1_jaccard_heatmap(agreement_data)
    figure2_coneness_vs_diversity(traces_df)
    figure3_raw_vs_centered(traces_df)
    figure4_alignment_manifold(manifold_df)
    figure4b_synthetic_baselines(manifold_df)
    figure5_trajectory_b09(manifold_df)
    figure6_center_of_gravity(manifold_df)
    figure7_model_stability(manifold_df)
    figure8_structured_vs_unstructured(manifold_df)
    figure_role_geometry()
    
    # Export Tableau data (for web animation)
    export_tableau_data(manifold_df)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)
    
    # Summary
    print("\nGenerated Figures:")
    for f in sorted(FIGURES_DIR.glob(f"*.{FIG_FORMAT}")):
        print(f"  ✓ {f.name}")
    
    print("\nRemaining:")
    print("  • Figure 0: Pipeline diagram (manual - draw.io)")
    print("  • Figure 5: Trajectory animation (use Tableau with exported CSV)")


if __name__ == "__main__":
    main()

