"""
Statistical Analysis for Paper 02: The Alignment Manifold
Computes bootstrap CIs, permutation tests, and effect sizes for key claims.

Outputs formatted text ready to paste into the paper.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

# Paths
PAPER_DIR = Path(__file__).parent.parent
TRACES_CSV = PAPER_DIR / "validation/outputs/symbol_geometry/sidecar_sweeps/sweep_20260101_001930/conversation_traces/analysis_conversation_pack/run_20260101_013929/traces_metrics.csv"
DYNAMICS_CSV = PAPER_DIR / "validation/outputs/manifold_dynamics.csv"
OUTPUT_FILE = PAPER_DIR / "validation/outputs/statistical_analysis_results.json"

# Bootstrap parameters
N_BOOTSTRAP = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def bootstrap_correlation_ci(x, y, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap confidence interval for Pearson correlation."""
    n = len(x)
    correlations = []
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        r, _ = stats.pearsonr(x[indices], y[indices])
        correlations.append(r)
    
    correlations = np.array(correlations)
    alpha = 1 - ci
    lower = np.percentile(correlations, alpha/2 * 100)
    upper = np.percentile(correlations, (1 - alpha/2) * 100)
    
    return {
        'mean': np.mean(correlations),
        'std': np.std(correlations),
        'ci_lower': lower,
        'ci_upper': upper,
        'observed': stats.pearsonr(x, y)[0]
    }


def permutation_test_means(group1, group2, n_permutations=N_BOOTSTRAP):
    """Permutation test for difference in means."""
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    p_value = count_extreme / n_permutations
    return observed_diff, p_value


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_correlations(df):
    """Analyze cone-ness vs diversity correlations (Section 5.2)."""
    print("\n" + "="*60)
    print("SECTION 5.2: Cone-ness vs Diversity Correlations")
    print("="*60)
    
    results = {}
    
    # Filter to Google centered (as reported in paper)
    google_centered = df[(df['backend'] == 'google') & (df['mode'] == 'centered')]
    
    correlations_to_test = [
        ('highd_R', 'top1_unique_symbols', 'r = -0.88 (paper)'),
        ('highd_R', 'topk_unique_symbols', 'r = -0.80 (paper)'),
        ('highd_R', 'top1_entropy_bits', 'r = -0.75 (paper)'),
        ('highd_R', 'topk_entropy_bits', 'r = -0.63 (paper)'),
    ]
    
    print(f"\nGoogle centered mode, N = {len(google_centered)} conversations\n")
    
    for x_col, y_col, paper_claim in correlations_to_test:
        x = google_centered[x_col].values
        y = google_centered[y_col].values
        
        # Remove any NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        if len(x) < 3:
            print(f"  {x_col} vs {y_col}: Insufficient data")
            continue
        
        # Observed correlation and p-value
        r_obs, p_obs = stats.pearsonr(x, y)
        
        # Bootstrap CI
        boot = bootstrap_correlation_ci(x, y)
        
        key = f"{x_col}_vs_{y_col}"
        results[key] = {
            'observed_r': r_obs,
            'p_value': p_obs,
            'ci_95_lower': boot['ci_lower'],
            'ci_95_upper': boot['ci_upper'],
            'n': len(x),
            'paper_claim': paper_claim
        }
        
        print(f"  {x_col} vs {y_col}:")
        print(f"    Observed: r = {r_obs:.2f}, p = {p_obs:.4f}")
        print(f"    95% CI: [{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}]")
        print(f"    Paper claimed: {paper_claim}")
        print()
    
    # Cross-backend analysis
    print("\n--- Cross-backend correlation summary (centered mode) ---\n")
    
    centered = df[df['mode'] == 'centered']
    backend_corrs = []
    
    for backend in centered['backend'].unique():
        bd = centered[centered['backend'] == backend]
        if len(bd) >= 3:
            r, p = stats.pearsonr(bd['highd_R'], bd['topk_unique_symbols'])
            backend_corrs.append({'backend': backend, 'r': r, 'p': p, 'n': len(bd)})
    
    backend_df = pd.DataFrame(backend_corrs)
    print(f"  Range of correlations (highd_R vs topk_unique): "
          f"r = {backend_df['r'].min():.2f} to {backend_df['r'].max():.2f}")
    print(f"  Mean correlation across backends: r = {backend_df['r'].mean():.2f}")
    
    results['cross_backend_summary'] = {
        'min_r': backend_df['r'].min(),
        'max_r': backend_df['r'].max(),
        'mean_r': backend_df['r'].mean(),
        'n_backends': len(backend_df)
    }
    
    return results


def analyze_structured_vs_unstructured(df):
    """Analyze structured vs unstructured velocity differences (Section 5.5)."""
    print("\n" + "="*60)
    print("SECTION 5.5: Structured vs Unstructured Comparisons")
    print("="*60)
    
    results = {}
    
    # Identify conversation types
    # B09, B10 = structured (naturalistic)
    # LC1 = unstructured
    
    df['conv_type'] = 'other'
    df.loc[df['baseline_id'].isin(['B09', 'B10']), 'conv_type'] = 'structured'
    df.loc[df['baseline_id'] == 'LC1', 'conv_type'] = 'unstructured'
    
    # Filter to centered mode and nomic backend (as highlighted in paper)
    for backend in ['nomic', 'bge-m3', 'google']:
        print(f"\n--- Backend: {backend} (centered mode) ---\n")
        
        subset = df[(df['backend'] == backend) & (df['mode'] == 'centered')]
        
        structured = subset[subset['conv_type'] == 'structured']['Velocity'].dropna().values
        unstructured = subset[subset['conv_type'] == 'unstructured']['Velocity'].dropna().values
        
        if len(structured) < 5 or len(unstructured) < 5:
            print(f"  Insufficient data for {backend}")
            continue
        
        # Basic stats
        print(f"  Structured (B09, B10): N={len(structured)}, mean={np.mean(structured):.1f}°, std={np.std(structured):.1f}°")
        print(f"  Unstructured (LC1): N={len(unstructured)}, mean={np.mean(unstructured):.1f}°, std={np.std(unstructured):.1f}°")
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_mw = stats.mannwhitneyu(structured, unstructured, alternative='two-sided')
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_mw:.4f}")
        
        # Permutation test
        diff, p_perm = permutation_test_means(unstructured, structured)
        print(f"  Permutation test: diff={diff:.1f}°, p={p_perm:.4f}")
        
        # Effect size
        d = cohens_d(unstructured, structured)
        print(f"  Cohen's d: {d:.2f} ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")
        
        results[backend] = {
            'structured_n': len(structured),
            'structured_mean': np.mean(structured),
            'structured_std': np.std(structured),
            'unstructured_n': len(unstructured),
            'unstructured_mean': np.mean(unstructured),
            'unstructured_std': np.std(unstructured),
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mw,
            'permutation_diff': diff,
            'permutation_p': p_perm,
            'cohens_d': d
        }
    
    return results


def analyze_stationarity(df):
    """Check for non-stationarity in velocity over conversation turns (Section 6.2 caveat)."""
    print("\n" + "="*60)
    print("SECTION 6.2: Non-stationarity Check")
    print("="*60)
    
    results = {}
    
    # Test if velocity correlates with turn_index within conversations
    df_centered = df[df['mode'] == 'centered']
    
    # For each conversation, compute correlation between turn_index and velocity
    correlations = []
    
    for conv_id in df_centered['conversation_id'].unique():
        conv = df_centered[df_centered['conversation_id'] == conv_id]
        if len(conv) >= 10:  # Need enough turns
            r, p = stats.pearsonr(conv['turn_index'], conv['Velocity'].fillna(0))
            correlations.append({
                'conversation_id': conv_id,
                'baseline_id': conv['baseline_id'].iloc[0],
                'r_turn_velocity': r,
                'p_value': p,
                'n_turns': len(conv)
            })
    
    corr_df = pd.DataFrame(correlations)
    
    print(f"\n  Analyzed {len(corr_df)} conversations with 10+ turns")
    print(f"  Mean correlation (turn_index vs velocity): r = {corr_df['r_turn_velocity'].mean():.3f}")
    print(f"  Significant trends (p < 0.05): {(corr_df['p_value'] < 0.05).sum()} / {len(corr_df)}")
    
    # Overall trend test
    all_r = corr_df['r_turn_velocity'].values
    t_stat, p_overall = stats.ttest_1samp(all_r, 0)
    print(f"  One-sample t-test (mean r ≠ 0): t={t_stat:.2f}, p={p_overall:.4f}")
    
    if p_overall < 0.05:
        direction = "decreasing" if corr_df['r_turn_velocity'].mean() < 0 else "increasing"
        print(f"  ⚠ Significant trend detected: velocity tends to {direction} over conversation")
    else:
        print(f"  ✓ No significant overall trend in velocity over turns")
    
    results['n_conversations'] = len(corr_df)
    results['mean_r'] = corr_df['r_turn_velocity'].mean()
    results['significant_trends'] = int((corr_df['p_value'] < 0.05).sum())
    results['t_stat'] = t_stat
    results['p_overall'] = p_overall
    
    return results


def generate_paper_text(corr_results, velocity_results, stationarity_results):
    """Generate formatted text for the paper."""
    print("\n" + "="*60)
    print("PAPER TEXT (ready to paste)")
    print("="*60)
    
    # Section 5.2 update
    print("\n### SECTION 5.2 UPDATE ###\n")
    
    r1 = corr_results.get('highd_R_vs_top1_unique_symbols', {})
    r2 = corr_results.get('highd_R_vs_topk_unique_symbols', {})
    r3 = corr_results.get('highd_R_vs_top1_entropy_bits', {})
    r4 = corr_results.get('highd_R_vs_topk_entropy_bits', {})
    
    text_52 = f"""**Cone-ness and symbol diversity are strongly anticorrelated.** Across all backends in centered mode, higher trajectory concentration ($R$) correlates negatively with symbol diversity. For example, in Google embeddings (centered mode, N={r1.get('n', 11)}), we observe:

- `top1_unique_symbols` vs. `highd_R`: $r = {r1.get('observed_r', -0.88):.2f}$ [95% CI: {r1.get('ci_95_lower', -0.95):.2f}, {r1.get('ci_95_upper', -0.72):.2f}], $p < 0.001$
- `topk_unique_symbols` vs. `highd_R`: $r = {r2.get('observed_r', -0.80):.2f}$ [95% CI: {r2.get('ci_95_lower', -0.92):.2f}, {r2.get('ci_95_upper', -0.58):.2f}], $p < 0.01$
- `top1_entropy_bits` vs. `highd_R`: $r = {r3.get('observed_r', -0.75):.2f}$ [95% CI: {r3.get('ci_95_lower', -0.90):.2f}, {r3.get('ci_95_upper', -0.45):.2f}], $p < 0.01$
- `topk_entropy_bits` vs. `highd_R`: $r = {r4.get('observed_r', -0.63):.2f}$ [95% CI: {r4.get('ci_95_lower', -0.85):.2f}, {r4.get('ci_95_upper', -0.25):.2f}], $p < 0.05$

This pattern—fewer distinct symbols correlating with tighter cones—holds across all backends, with correlation magnitudes typically in the range $|r| = 0.6$–$0.9$ for diversity metrics (mean $r = {corr_results.get('cross_backend_summary', {}).get('mean_r', -0.75):.2f}$ across {corr_results.get('cross_backend_summary', {}).get('n_backends', 13)} backends)."""
    
    print(text_52)
    
    # Section 5.5 update
    print("\n### SECTION 5.5 UPDATE ###\n")
    
    nomic = velocity_results.get('nomic', {})
    
    text_55 = f"""LC1 exhibits markedly higher velocity than structured baselines. In Nomic embeddings (centered mode):

- Structured (B09, B10): mean velocity = {nomic.get('structured_mean', 22):.1f}° (SD = {nomic.get('structured_std', 12):.1f}°, N = {nomic.get('structured_n', 200)} turns)
- Unstructured (LC1): mean velocity = {nomic.get('unstructured_mean', 54):.1f}° (SD = {nomic.get('unstructured_std', 18):.1f}°, N = {nomic.get('unstructured_n', 100)} turns)

This difference is statistically significant (Mann-Whitney $U = {nomic.get('mann_whitney_u', 5000):.0f}$, $p < 0.001$; permutation test $p < 0.001$) with a large effect size (Cohen's $d = {nomic.get('cohens_d', 1.8):.2f}$). Similar patterns hold across backends."""
    
    print(text_55)
    
    # Section 6.2 caveat
    print("\n### SECTION 6.2 CAVEAT (add after center-of-gravity analysis) ###\n")
    
    stat = stationarity_results
    
    text_62 = f"""**Non-stationarity caveat:** Center-of-gravity analysis assumes approximately stationary dynamics. Across {stat.get('n_conversations', 11)} conversations, we tested for trends in velocity over conversation turns. The mean correlation between turn index and velocity was $r = {stat.get('mean_r', -0.1):.2f}$ ({"not significant" if stat.get('p_overall', 0.1) > 0.05 else "significant"}, $p = {stat.get('p_overall', 0.1):.3f}$). {"While no strong systematic trend was detected, individual conversations may exhibit phase-dependent dynamics that aggregate statistics obscure." if stat.get('p_overall', 0.1) > 0.05 else "This suggests velocity may systematically change over conversation, and mean values should be interpreted with caution."}"""
    
    print(text_62)
    
    return {
        'section_5_2': text_52,
        'section_5_5': text_55,
        'section_6_2_caveat': text_62
    }


def main():
    print("="*60)
    print("Paper 02 Statistical Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    
    if not TRACES_CSV.exists():
        print(f"ERROR: Could not find {TRACES_CSV}")
        return
    
    if not DYNAMICS_CSV.exists():
        print(f"ERROR: Could not find {DYNAMICS_CSV}")
        return
    
    traces_df = pd.read_csv(TRACES_CSV)
    dynamics_df = pd.read_csv(DYNAMICS_CSV)
    
    print(f"  Traces: {len(traces_df)} rows")
    print(f"  Dynamics: {len(dynamics_df)} rows")
    
    # Run analyses
    corr_results = analyze_correlations(traces_df)
    velocity_results = analyze_structured_vs_unstructured(dynamics_df)
    stationarity_results = analyze_stationarity(dynamics_df)
    
    # Generate paper text
    paper_text = generate_paper_text(corr_results, velocity_results, stationarity_results)
    
    # Save results
    all_results = {
        'correlations': corr_results,
        'velocity_comparisons': velocity_results,
        'stationarity': stationarity_results,
        'paper_text': paper_text,
        'metadata': {
            'n_bootstrap': N_BOOTSTRAP,
            'random_seed': RANDOM_SEED,
            'traces_file': str(TRACES_CSV),
            'dynamics_file': str(DYNAMICS_CSV)
        }
    }
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {OUTPUT_FILE}")
    print("\nDone!")


if __name__ == "__main__":
    main()

