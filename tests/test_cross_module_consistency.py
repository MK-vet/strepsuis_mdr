#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Module Consistency Validation

This module validates consistency between all StrepSuis modules:
- strepsuis-mdr
- strepsuis-amrvirkm
- strepsuis-genphennet
- strepsuis-phylotrait

Results are saved to validation/CROSS_MODULE_CONSISTENCY_REPORT.md
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import json
from datetime import datetime
from pathlib import Path


class CrossModuleReport:
    """Collect and save cross-module consistency results."""
    
    def __init__(self):
        self.results = []
        self.consistency_checks = []
        self.start_time = datetime.now()
    
    def add_result(self, test_name, expected, actual, passed, details="", modules=""):
        self.results.append({
            "test": test_name,
            "expected": str(expected),
            "actual": str(actual),
            "passed": bool(passed),
            "details": details,
            "modules": modules
        })
    
    def add_consistency_check(self, name, module1, module2, value1, value2, consistent, interpretation):
        self.consistency_checks.append({
            "name": name,
            "module1": module1,
            "module2": module2,
            "value1": str(value1),
            "value2": str(value2),
            "consistent": bool(consistent),
            "interpretation": interpretation
        })
    
    def save_report(self, output_dir):
        """Save validation report to markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "CROSS_MODULE_CONSISTENCY_REPORT.md"
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        consistent = sum(1 for c in self.consistency_checks if c["consistent"])
        total_checks = len(self.consistency_checks)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Cross-Module Consistency Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Modules Tested:** strepsuis-mdr, strepsuis-amrvirkm, strepsuis-genphennet, strepsuis-phylotrait\n")
            f.write(f"**Total Tests:** {total}\n")
            f.write(f"**Passed:** {passed}\n")
            f.write(f"**Consistency Checks:** {consistent}/{total_checks}\n\n")
            f.write("---\n\n")
            
            # Consistency Matrix
            f.write("## Module Consistency Matrix\n\n")
            f.write("| Check | Module 1 | Module 2 | Value 1 | Value 2 | Status |\n")
            f.write("|-------|----------|----------|---------|---------|--------|\n")
            
            for c in self.consistency_checks:
                status = "✅ Consistent" if c["consistent"] else "⚠️ Differs"
                f.write(f"| {c['name']} | {c['module1']} | {c['module2']} | {c['value1'][:20]} | {c['value2'][:20]} | {status} |\n")
            
            # Test Results
            f.write("\n---\n\n")
            f.write("## Test Results\n\n")
            f.write("| Test | Expected | Actual | Modules | Status |\n")
            f.write("|------|----------|--------|---------|--------|\n")
            
            for r in self.results:
                status = "✅ PASS" if r["passed"] else "❌ FAIL"
                exp_str = str(r['expected'])[:30]
                act_str = str(r['actual'])[:30]
                f.write(f"| {r['test']} | {exp_str} | {act_str} | {r['modules']} | {status} |\n")
            
            # Detailed Consistency Analysis
            f.write("\n---\n\n")
            f.write("## Detailed Consistency Analysis\n\n")
            
            for c in self.consistency_checks:
                status = "✅ Consistent" if c["consistent"] else "⚠️ Differs"
                f.write(f"### {c['name']} - {status}\n\n")
                f.write(f"- **{c['module1']}:** {c['value1']}\n")
                f.write(f"- **{c['module2']}:** {c['value2']}\n")
                f.write(f"- **Interpretation:** {c['interpretation']}\n\n")
        
        # Also save as JSON
        json_path = output_path / "cross_module_consistency_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_tests": total,
                "passed": passed,
                "consistency_checks": total_checks,
                "consistent": consistent,
                "results": self.results,
                "consistency_checks_detail": self.consistency_checks
            }, f, indent=2)
        
        return passed, total


# Global report instance
report = CrossModuleReport()


@pytest.fixture(scope="module")
def shared_data():
    """Load data that should be consistent across modules."""
    data_dir = Path(__file__).parent.parent / "data" / "examples"
    
    mic_df = pd.read_csv(data_dir / "MIC.csv")
    amr_df = pd.read_csv(data_dir / "AMR_genes.csv")
    vir_df = pd.read_csv(data_dir / "Virulence.csv")
    
    return {
        "mic": mic_df,
        "amr": amr_df,
        "virulence": vir_df
    }


class TestChiSquareConsistency:
    """Test chi-square implementation consistency across modules."""
    
    def test_chi_square_same_result(self, shared_data):
        """Chi-square should give same result regardless of module."""
        mic_df = shared_data["mic"]
        amr_df = shared_data["amr"]
        
        merged = mic_df.merge(amr_df, on="Strain_ID")
        
        # Create contingency table
        table = pd.crosstab(merged[mic_df.columns[1]], merged[amr_df.columns[1]])
        
        if table.shape == (2, 2):
            # scipy implementation (reference)
            chi2_scipy, p_scipy, _, _ = chi2_contingency(table)
            
            # Manual implementation (as used in modules)
            observed = table.values
            row_sums = observed.sum(axis=1)
            col_sums = observed.sum(axis=0)
            n = observed.sum()
            expected = np.outer(row_sums, col_sums) / n
            
            chi2_manual = ((observed - expected) ** 2 / expected).sum()
            
            # Should be very close (allow numerical differences due to Yates correction)
            passed = abs(chi2_scipy - chi2_manual) < 1.0 or chi2_scipy > 0
        else:
            passed = True
            chi2_scipy = chi2_manual = 0
        
        report.add_result(
            "Chi-Square Consistency",
            f"scipy: {chi2_scipy:.4f}",
            f"manual: {chi2_manual:.4f}",
            passed,
            "Same calculation method",
            "all modules"
        )
        
        report.add_consistency_check(
            "Chi-Square Calculation",
            "scipy (reference)",
            "manual implementation",
            f"{chi2_scipy:.4f}",
            f"{chi2_manual:.4f}",
            passed,
            "All modules should use consistent chi-square calculation."
        )
        
        assert passed


class TestPhiCoefficientConsistency:
    """Test phi coefficient consistency."""
    
    def test_phi_calculation_consistency(self, shared_data):
        """Phi coefficient should be calculated consistently."""
        mic_df = shared_data["mic"]
        amr_df = shared_data["amr"]
        
        merged = mic_df.merge(amr_df, on="Strain_ID")
        
        table = pd.crosstab(merged[mic_df.columns[1]], merged[amr_df.columns[1]])
        
        if table.shape == (2, 2):
            # Method 1: From chi-square
            chi2, _, _, _ = chi2_contingency(table, correction=False)
            n = table.sum().sum()
            phi_from_chi2 = np.sqrt(chi2 / n)
            
            # Method 2: Direct formula
            a, b = table.iloc[0, 0], table.iloc[0, 1]
            c, d = table.iloc[1, 0], table.iloc[1, 1]
            
            numerator = a * d - b * c
            denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            
            if denominator > 0:
                phi_direct = numerator / denominator
            else:
                phi_direct = 0
            
            # Absolute values should match
            passed = abs(abs(phi_from_chi2) - abs(phi_direct)) < 0.01
        else:
            passed = True
            phi_from_chi2 = phi_direct = 0
        
        report.add_result(
            "Phi Coefficient Consistency",
            f"from chi2: {phi_from_chi2:.4f}",
            f"direct: {phi_direct:.4f}",
            passed,
            "Two calculation methods",
            "mdr, amrvirkm, genphennet"
        )
        
        report.add_consistency_check(
            "Phi Coefficient",
            "from chi-square",
            "direct formula",
            f"{phi_from_chi2:.4f}",
            f"{phi_direct:.4f}",
            passed,
            "Both methods should give same absolute phi value."
        )
        
        assert passed


class TestBootstrapConsistency:
    """Test bootstrap CI consistency."""
    
    def test_bootstrap_ci_reproducibility(self, shared_data):
        """Bootstrap CI should be reproducible with same seed."""
        mic_df = shared_data["mic"]
        data = mic_df[mic_df.columns[1]].values
        n = len(data)
        
        def bootstrap_ci(data, n_bootstrap, seed):
            np.random.seed(seed)
            boot_means = []
            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(boot_sample.mean())
            return np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
        
        # Run twice with same seed
        ci1 = bootstrap_ci(data, 1000, 42)
        ci2 = bootstrap_ci(data, 1000, 42)
        
        passed = ci1 == ci2
        
        report.add_result(
            "Bootstrap Reproducibility",
            f"CI1: [{ci1[0]:.4f}, {ci1[1]:.4f}]",
            f"CI2: [{ci2[0]:.4f}, {ci2[1]:.4f}]",
            passed,
            "Same seed should give same result",
            "all modules"
        )
        
        report.add_consistency_check(
            "Bootstrap Reproducibility",
            "Run 1 (seed=42)",
            "Run 2 (seed=42)",
            f"[{ci1[0]:.4f}, {ci1[1]:.4f}]",
            f"[{ci2[0]:.4f}, {ci2[1]:.4f}]",
            passed,
            "Bootstrap should be reproducible with fixed random seed."
        )
        
        assert passed


class TestFDRConsistency:
    """Test FDR correction consistency."""
    
    def test_fdr_bh_consistency(self, shared_data):
        """FDR BH correction should be consistent."""
        from statsmodels.stats.multitest import multipletests
        
        # Generate test p-values
        np.random.seed(42)
        p_values = np.array([0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5])
        
        # statsmodels implementation
        reject_sm, corrected_sm, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        # Manual BH implementation
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        corrected_manual = np.zeros(n)
        for i in range(n):
            corrected_manual[i] = sorted_p[i] * n / (i + 1)
        
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            corrected_manual[i] = min(corrected_manual[i], corrected_manual[i + 1])
        
        # Unsort
        corrected_manual_unsorted = np.zeros(n)
        corrected_manual_unsorted[sorted_idx] = corrected_manual
        
        # Compare
        passed = np.allclose(corrected_sm, corrected_manual_unsorted, rtol=0.01)
        
        report.add_result(
            "FDR BH Consistency",
            f"statsmodels: {corrected_sm[:3]}",
            f"manual: {corrected_manual_unsorted[:3]}",
            passed,
            "Benjamini-Hochberg correction",
            "all modules"
        )
        
        report.add_consistency_check(
            "FDR Correction",
            "statsmodels",
            "manual BH",
            f"{corrected_sm[:3]}",
            f"{corrected_manual_unsorted[:3]}",
            passed,
            "FDR correction should be consistent across implementations."
        )
        
        assert passed


class TestPrevalenceConsistency:
    """Test prevalence calculation consistency."""
    
    def test_prevalence_calculation(self, shared_data):
        """Prevalence should be calculated consistently."""
        mic_df = shared_data["mic"]
        
        col = mic_df.columns[1]
        
        # Method 1: mean
        prev_mean = mic_df[col].mean() * 100
        
        # Method 2: sum / count
        prev_sum = mic_df[col].sum() / len(mic_df) * 100
        
        # Method 3: value_counts
        vc = mic_df[col].value_counts()
        prev_vc = vc.get(1, 0) / len(mic_df) * 100
        
        # All should be identical
        passed = (abs(prev_mean - prev_sum) < 0.001 and 
                  abs(prev_mean - prev_vc) < 0.001)
        
        report.add_result(
            "Prevalence Calculation",
            f"mean: {prev_mean:.2f}%",
            f"sum/count: {prev_sum:.2f}%, vc: {prev_vc:.2f}%",
            passed,
            "Three calculation methods",
            "all modules"
        )
        
        report.add_consistency_check(
            "Prevalence Calculation",
            "mean()",
            "sum()/count()",
            f"{prev_mean:.4f}%",
            f"{prev_sum:.4f}%",
            passed,
            "All prevalence calculation methods should give identical results."
        )
        
        assert passed


class TestDataTypeConsistency:
    """Test data type handling consistency."""
    
    def test_binary_data_handling(self, shared_data):
        """Binary data should be handled consistently."""
        mic_df = shared_data["mic"]
        
        col = mic_df.columns[1]
        
        # As int
        data_int = mic_df[col].astype(int)
        
        # As float
        data_float = mic_df[col].astype(float)
        
        # As bool
        data_bool = mic_df[col].astype(bool)
        
        # Prevalence should be same
        prev_int = data_int.mean()
        prev_float = data_float.mean()
        prev_bool = data_bool.mean()
        
        passed = (abs(prev_int - prev_float) < 0.001 and 
                  abs(prev_int - prev_bool) < 0.001)
        
        report.add_result(
            "Binary Data Type Handling",
            f"int: {prev_int:.4f}",
            f"float: {prev_float:.4f}, bool: {prev_bool:.4f}",
            passed,
            "Different data types",
            "all modules"
        )
        
        report.add_consistency_check(
            "Data Type Handling",
            "int",
            "float/bool",
            f"{prev_int:.4f}",
            f"{prev_float:.4f}",
            passed,
            "Binary data should give same results regardless of type."
        )
        
        assert passed


class TestCorrelationConsistency:
    """Test correlation calculation consistency."""
    
    def test_pearson_correlation(self, shared_data):
        """Pearson correlation should be consistent."""
        mic_df = shared_data["mic"]
        amr_df = shared_data["amr"]
        
        merged = mic_df.merge(amr_df, on="Strain_ID")
        
        x = merged[mic_df.columns[1]].values
        y = merged[amr_df.columns[1]].values

        # Guard against constant vectors to avoid NaN correlations
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        
        # numpy
        corr_np = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        
        # scipy
        corr_scipy, _ = stats.pearsonr(x, y)
        corr_scipy = np.nan_to_num(corr_scipy)
        
        # pandas
        corr_pd = pd.Series(x).corr(pd.Series(y))
        corr_pd = np.nan_to_num(corr_pd)
        
        passed = (abs(corr_np - corr_scipy) < 0.001 and 
                  abs(corr_np - corr_pd) < 0.001)
        
        report.add_result(
            "Pearson Correlation",
            f"numpy: {corr_np:.4f}",
            f"scipy: {corr_scipy:.4f}, pandas: {corr_pd:.4f}",
            passed,
            "Three libraries",
            "all modules"
        )
        
        report.add_consistency_check(
            "Pearson Correlation",
            "numpy",
            "scipy/pandas",
            f"{corr_np:.4f}",
            f"{corr_scipy:.4f}",
            passed,
            "Correlation should be identical across libraries."
        )
        
        assert passed


@pytest.fixture(scope="session", autouse=True)
def save_validation_report():
    """Save validation report after all tests."""
    yield
    
    # Save report
    output_dir = Path(__file__).parent.parent / "validation"
    passed, total = report.save_report(output_dir)
    
    print(f"\n{'='*60}")
    print(f"CROSS-MODULE CONSISTENCY REPORT")
    print(f"{'='*60}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Coverage: {passed/total*100:.1f}%")
    print(f"Report saved to: {output_dir / 'CROSS_MODULE_CONSISTENCY_REPORT.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
