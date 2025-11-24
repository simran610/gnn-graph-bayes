"""
Statistical Significance Testing for Benchmark Results
========================================================
Run this AFTER benchmark_unified.py to test if your GNN's improvements
are statistically significant.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


class BenchmarkStatistics:
    def __init__(self, results_dir="benchmark_unified_results"):
        self.results_dir = Path(results_dir)
        self.summary = pd.read_csv(self.results_dir / "summary.csv")
        
        # Filter out failed methods (infinite MAE)
        self.summary = self.summary[self.summary['MAE'] != float('inf')].copy()
        
        print(f"✓ Loaded {len(self.summary)} valid results")
        print(f"  Methods: {self.summary['Method'].unique().tolist()}")
        print(f"  Networks: {len(self.summary['Network'].unique())}")
    
    def test_accuracy_difference(self, method1="GNN (Ours)", method2="VE-Exact", 
                                 metric="MAE", alpha=0.05):
        """
        Test if method1 has significantly different accuracy than method2
        
        H0: method1 and method2 have same accuracy
        H1: method1 and method2 have different accuracy
        """
        print(f"\n{'='*70}")
        print(f"ACCURACY TEST: {method1} vs {method2} ({metric})")
        print(f"{'='*70}")
        
        data1 = self.summary[self.summary['Method'] == method1][metric]
        data2 = self.summary[self.summary['Method'] == method2][metric]
        
        if len(data1) == 0 or len(data2) == 0:
            print(f"⚠ Missing data for one or both methods")
            return
        
        # Descriptive stats
        print(f"\n{method1}:")
        print(f"  Mean: {data1.mean():.4f} ± {data1.std():.4f}")
        print(f"  Median: {data1.median():.4f}")
        print(f"  Range: [{data1.min():.4f}, {data1.max():.4f}]")
        
        print(f"\n{method2}:")
        print(f"  Mean: {data2.mean():.4f} ± {data2.std():.4f}")
        print(f"  Median: {data2.median():.4f}")
        print(f"  Range: [{data2.min():.4f}, {data2.max():.4f}]")
        
        # Test normality
        _, p_norm1 = stats.shapiro(data1)
        _, p_norm2 = stats.shapiro(data2)
        
        print(f"\nNormality tests:")
        print(f"  {method1}: p={p_norm1:.4f} {'✓ Normal' if p_norm1 > 0.05 else '✗ Non-normal'}")
        print(f"  {method2}: p={p_norm2:.4f} {'✓ Normal' if p_norm2 > 0.05 else '✗ Non-normal'}")
        
        # Choose test based on normality
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Both normal → paired t-test
            stat, p_value = stats.ttest_rel(data1, data2)
            test_name = "Paired t-test"
        else:
            # Non-normal → Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(data1, data2)
            test_name = "Wilcoxon signed-rank test"
        
        print(f"\n{test_name}:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < alpha:
            print(f"  ✓ SIGNIFICANT at α={alpha} (reject H0)")
            diff = data1.mean() - data2.mean()
            if diff < 0:
                print(f"  → {method1} has BETTER (lower) {metric}")
            else:
                print(f"  → {method2} has BETTER (lower) {metric}")
        else:
            print(f"  ✗ NOT significant (fail to reject H0)")
            print(f"  → No evidence of difference in {metric}")
        
        # Effect size (Cohen's d)
        mean_diff = data1.mean() - data2.mean()
        pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
        cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        print(f"\nEffect size (Cohen's d): {cohen_d:.4f}")
        if abs(cohen_d) < 0.2:
            print(f"  → Small effect")
        elif abs(cohen_d) < 0.5:
            print(f"  → Medium effect")
        else:
            print(f"  → Large effect")
        
        return {
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cohens_d': cohen_d
        }
    
    def test_speed_difference(self, method1="GNN (Ours)", method2="VE-Exact", 
                             alpha=0.05):
        """
        Test if method1 is significantly faster than method2
        
        H0: method1 ≥ method2 (method1 is not faster)
        H1: method1 < method2 (method1 is faster)
        """
        print(f"\n{'='*70}")
        print(f"SPEED TEST: {method1} vs {method2}")
        print(f"{'='*70}")
        
        data1 = self.summary[self.summary['Method'] == method1]['Time_ms']
        data2 = self.summary[self.summary['Method'] == method2]['Time_ms']
        
        if len(data1) == 0 or len(data2) == 0:
            print(f"⚠ Missing data for one or both methods")
            return
        
        # Descriptive stats
        print(f"\n{method1}:")
        print(f"  Mean: {data1.mean():.2f}ms ± {data1.std():.2f}ms")
        print(f"  Median: {data1.median():.2f}ms")
        
        print(f"\n{method2}:")
        print(f"  Mean: {data2.mean():.2f}ms ± {data2.std():.2f}ms")
        print(f"  Median: {data2.median():.2f}ms")
        
        speedup = data2.mean() / data1.mean()
        print(f"\nSpeedup: {speedup:.2f}×")
        
        # One-sided test (is method1 faster?)
        if np.all(data1 == data2):
            print("\n✗ Data is identical, no test needed")
            return
        
        # Use Mann-Whitney U test (one-sided, non-parametric)
        stat, p_value_two = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        _, p_value_one = stats.mannwhitneyu(data1, data2, alternative='less')
        
        print(f"\nMann-Whitney U test (one-sided):")
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value (one-sided): {p_value_one:.6f}")
        print(f"  p-value (two-sided): {p_value_two:.6f}")
        
        if p_value_one < alpha:
            print(f"  ✓ SIGNIFICANT at α={alpha}")
            print(f"  → {method1} is SIGNIFICANTLY FASTER than {method2}")
            print(f"  → Can claim: '{speedup:.1f}× speedup (p<{alpha})'")
        else:
            print(f"  ✗ NOT significant")
            print(f"  → Cannot claim {method1} is faster")
        
        return {
            'test': 'Mann-Whitney U (one-sided)',
            'statistic': stat,
            'p_value': p_value_one,
            'significant': p_value_one < alpha,
            'speedup': speedup
        }
    
    def compare_all_methods(self, metric="MAE", alpha=0.05):
        """
        Pairwise comparison of all methods using ANOVA or Kruskal-Wallis
        """
        print(f"\n{'='*70}")
        print(f"ALL METHODS COMPARISON ({metric})")
        print(f"{'='*70}")
        
        methods = self.summary['Method'].unique()
        
        # Prepare data for each method
        data_by_method = []
        for method in methods:
            data = self.summary[self.summary['Method'] == method][metric].values
            data_by_method.append(data)
        
        # Test if all groups have same distribution
        stat, p_value = stats.kruskal(*data_by_method)
        
        print(f"\nKruskal-Wallis H-test:")
        print(f"  H-statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < alpha:
            print(f"  ✓ SIGNIFICANT at α={alpha}")
            print(f"  → At least one method is different")
            print(f"\n  Running post-hoc pairwise tests...")
            self._posthoc_tests(metric, alpha)
        else:
            print(f"  ✗ NOT significant")
            print(f"  → No evidence that methods differ")
    
    def _posthoc_tests(self, metric="MAE", alpha=0.05):
        """Bonferroni-corrected pairwise tests"""
        methods = self.summary['Method'].unique().tolist()
        n_comparisons = len(methods) * (len(methods) - 1) // 2
        alpha_corrected = alpha / n_comparisons  # Bonferroni correction
        
        print(f"\n  Bonferroni-corrected α: {alpha_corrected:.6f}")
        print(f"  ({n_comparisons} pairwise comparisons)")
        
        results = []
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                data1 = self.summary[self.summary['Method'] == m1][metric].values
                data2 = self.summary[self.summary['Method'] == m2][metric].values
                
                _, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                significant = "✓" if p < alpha_corrected else "✗"
                results.append({
                    'Method 1': m1,
                    'Method 2': m2,
                    'p-value': p,
                    'Significant': significant
                })
        
        df_results = pd.DataFrame(results)
        print("\n", df_results.to_string(index=False))
        
        # Save
        df_results.to_csv(self.results_dir / "statistical_tests.csv", index=False)
        print(f"\n  ✓ Saved to {self.results_dir}/statistical_tests.csv")
    
    def visualize_distributions(self, metric="MAE"):
        """Create violin plots for visual comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # MAE distribution
        methods = self.summary['Method'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        parts = axes[0].violinplot(
            [self.summary[self.summary['Method']==m][metric].values for m in methods],
            showmeans=True,
            showmedians=True
        )
        axes[0].set_xticks(range(1, len(methods)+1))
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].set_ylabel(metric)
        axes[0].set_title(f'{metric} Distribution by Method')
        axes[0].grid(True, alpha=0.3)
        
        # Timing distribution (log scale)
        parts = axes[1].violinplot(
            [self.summary[self.summary['Method']==m]['Time_ms'].values for m in methods],
            showmeans=True,
            showmedians=True
        )
        axes[1].set_xticks(range(1, len(methods)+1))
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1].set_ylabel('Time (ms, log scale)')
        axes[1].set_yscale('log')
        axes[1].set_title('Inference Time Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'statistical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved violin plots to {self.results_dir}/statistical_distributions.png")
    
    def generate_paper_summary(self):
        """Generate summary statistics for paper"""
        print(f"\n{'='*70}")
        print("PAPER-READY SUMMARY")
        print(f"{'='*70}")
        
        methods = self.summary['Method'].unique()
        
        summary = []
        for method in methods:
            data = self.summary[self.summary['Method'] == method]
            
            summary.append({
                'Method': method,
                'MAE (mean±std)': f"{data['MAE'].mean():.4f}±{data['MAE'].std():.4f}",
                'RMSE (mean)': f"{data['RMSE'].mean():.4f}",
                'Time (mean±std, ms)': f"{data['Time_ms'].mean():.2f}±{data['Time_ms'].std():.2f}",
                'Success Rate': f"{data['Success_Rate'].mean():.1%}"
            })
        
        df_summary = pd.DataFrame(summary)
        print("\n", df_summary.to_string(index=False))
        
        # LaTeX table
        print(f"\n{'='*70}")
        print("LATEX TABLE (copy this to your paper)")
        print(f"{'='*70}")
        print()
        print(df_summary.to_latex(index=False, float_format="%.4f"))
        
        # Save
        df_summary.to_csv(self.results_dir / "paper_summary.csv", index=False)
        print(f"✓ Saved to {self.results_dir}/paper_summary.csv")


def main():
    print("="*80)
    print("STATISTICAL ANALYSIS OF BENCHMARK RESULTS")
    print("="*80)
    
    try:
        stats_analyzer = BenchmarkStatistics("benchmark_unified_results")
    except FileNotFoundError:
        print("\n⚠ ERROR: benchmark_unified_results/summary.csv not found")
        print("   Please run benchmark_unified.py first!")
        return
    
    # Test 1: Accuracy comparison (GNN vs VE-Exact)
    stats_analyzer.test_accuracy_difference(
        method1="GNN (Ours)",
        method2="VE-Exact",
        metric="MAE"
    )
    
    # Test 2: Speed comparison (GNN vs VE-Exact)
    stats_analyzer.test_speed_difference(
        method1="GNN (Ours)",
        method2="VE-Exact"
    )
    
    # Test 3: Speed comparison (GNN vs BP)
    stats_analyzer.test_speed_difference(
        method1="GNN (Ours)",
        method2="BP-Approx"
    )
    
    # Test 4: All methods comparison
    stats_analyzer.compare_all_methods(metric="MAE")
    
    # Visualize
    stats_analyzer.visualize_distributions(metric="MAE")
    
    # Paper summary
    stats_analyzer.generate_paper_summary()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nHow to use these results in your paper:")
    print("1. If p < 0.05: Claim 'statistically significant improvement'")
    print("2. Report effect size (Cohen's d) for magnitude of difference")
    print("3. Use violin plots to show distribution differences")
    print("4. Copy LaTeX table directly to paper")
    print("\nExample claim:")
    print("  'Our GNN achieves X.XX MAE, significantly outperforming")
    print("   Variable Elimination (p<0.05, Cohen's d=X.XX)'")


if __name__ == "__main__":
    main()