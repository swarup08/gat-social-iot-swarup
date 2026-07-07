"""
====================================================
Paired Comparison vs Random (Step 3 follow-up)
Project : GAT-based Secure Social IoT Framework
====================================================

The pooled mean +/- std table (multi_graph_summary.csv) can't
distinguish most methods from each other because graph-to-graph
topology variance (std ~0.15-0.17) dominates the between-method
differences (~0.05-0.06). This re-analyzes the same 40-graph raw
results (results/tables/multi_graph_raw_results.csv, no retraining)
as a PAIRED comparison: for each graph individually, method's
infection minus Random's infection on that same graph. This controls
for graph-to-graph variance and is a more powerful test than the
pooled comparison.

Reports, per method vs Random:
- mean paired difference (negative = method beats Random on average)
- paired t-test p-value
- Wilcoxon signed-rank p-value
- Shapiro-Wilk p-value on the paired differences, to indicate which
  test's assumptions are better satisfied (t-test assumes normally
  distributed differences; Wilcoxon does not)

No interpretation/conclusion here by design -- just the numbers.
"""

import pandas as pd
from scipy import stats

RAW_RESULTS_PATH = "results/tables/multi_graph_raw_results.csv"
OUTPUT_PATH = "results/tables/paired_comparison_vs_random.csv"


def run_paired_comparison():

    df = pd.read_csv(RAW_RESULTS_PATH)

    # Wide: rows = seed (graph instance), columns = method, values = infection
    infection_wide = df.pivot(index="seed", columns="method", values="infection")

    random_infection = infection_wide["Random"]

    methods = [m for m in infection_wide.columns if m != "Random"]

    rows = []

    for method in methods:

        method_infection = infection_wide[method]

        paired_diff = method_infection - random_infection

        t_stat, t_pvalue = stats.ttest_rel(method_infection, random_infection)

        # Wilcoxon is undefined if all differences are zero; guard just in case.
        if (paired_diff != 0).any():
            w_stat, w_pvalue = stats.wilcoxon(method_infection, random_infection)
        else:
            w_stat, w_pvalue = float("nan"), float("nan")

        shapiro_stat, shapiro_pvalue = stats.shapiro(paired_diff)

        rows.append({
            "method": method,
            "n_graphs": len(paired_diff),
            "mean_paired_diff_vs_random": paired_diff.mean(),
            "std_paired_diff_vs_random": paired_diff.std(),
            "t_test_statistic": t_stat,
            "t_test_pvalue": t_pvalue,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_pvalue,
            "shapiro_normality_pvalue": shapiro_pvalue,
        })

    result = pd.DataFrame(rows)

    method_order = [
        "No Pruning", "Threshold", "Top-K",
        "Degree Centrality", "Betweenness Centrality", "Dynamic RL",
    ]
    result["method"] = pd.Categorical(result["method"], categories=method_order, ordered=True)
    result = result.sort_values("method").reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 100)
    print("Paired Comparison vs Random (infection ratio, per-graph paired difference)")
    print("=" * 100)
    print(result.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    run_paired_comparison()
