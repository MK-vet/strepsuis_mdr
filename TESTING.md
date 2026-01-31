# Testing Guide for StrepSuisMDR

This document describes how to run tests locally and in CI/CD for the StrepSuisMDR package.

## Overview

The test suite includes:
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows with real data
- **Workflow tests**: End-to-end validation of analysis pipelines
- **End-to-end tests**: Full pipeline tests using real example datasets

## Test Data Strategy

### Example Data Location
Real example data is located in `examples/` directory:
- `MIC.csv` - Minimum Inhibitory Concentration data (92 strains)
- `AMR_genes.csv` - Antimicrobial resistance gene profiles (92 strains)
- `Virulence.csv` - Virulence factor profiles (92 strains)
- `MLST.csv`, `Serotype.csv`, `MGE.csv`, `Plasmid.csv` - Additional metadata

### Mini Datasets for CI
For fast CI execution, tests automatically create mini datasets:
- **Mini dataset**: First 10 strains from example data
- **Execution time**: <5 seconds for full test suite
- **Purpose**: Fast validation in GitHub Actions

### Full Datasets for Local Testing
Full example datasets are used for comprehensive validation:
- **Full dataset**: All 92 strains
- **Execution time**: 30-60 seconds
- **Purpose**: Complete validation before releases
- **Marked as**: `@pytest.mark.slow` (skipped in CI)

## Quick Start

### Install Development Dependencies

```bash
pip install -e .[dev]
```

### Run All Tests

```bash
pytest
```

### Run with Coverage Report

```bash
pytest --cov --cov-report=term --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the detailed coverage report.

## Test Categories

### Unit Tests
Fast tests that validate individual components:

```bash
pytest -m unit -v
```

### Integration Tests
Tests that use real example data:

```bash
pytest -m integration -v
```

### Workflow Tests
Complete end-to-end pipeline tests (may be slower):

```bash
pytest tests/test_workflow.py -v
```

### Exclude Slow Tests
For rapid iteration during development:

```bash
pytest -m "not slow" -v
```

### End-to-End Tests
Comprehensive tests using real example data:

```bash
# Fast tests with mini datasets (for CI)
pytest tests/test_end_to_end.py -m "integration and not slow" -v

# Full tests with complete datasets (local only)
pytest tests/test_end_to_end.py -v
```

## Running Specific Test Files

```bash
# Test basic functionality
pytest tests/test_basic.py -v

# Test configuration
pytest tests/test_config.py -v

# Test CLI interface
pytest tests/test_cli.py -v

# Test analyzer functionality
pytest tests/test_analyzer.py -v

# Test data validation
pytest tests/test_data_validation.py -v

# Test complete workflows
pytest tests/test_workflow.py -v

# Test end-to-end pipelines (NEW!)
pytest tests/test_end_to_end.py -v
```

## Coverage Requirements

- **Target Coverage**: 60% minimum for CI/CD
- **Recommended Coverage**: 80%+ for production code

### Generate Coverage Report

```bash
# Terminal report
pytest --cov --cov-report=term-missing

# HTML report
pytest --cov --cov-report=html
open htmlcov/index.html

# XML report (for CI tools)
pytest --cov --cov-report=xml
```

### Coverage by Module

Check coverage for specific modules:

```bash
pytest --cov=strepsuis_mdr --cov-report=term-missing
```

## Local Development Workflow

### 1. Quick Check (Fast Tests Only)
```bash
pytest -m "not slow" --cov-fail-under=0
```

### 2. Full Test Suite
```bash
pytest --cov
```

### 3. Pre-Commit Validation
```bash
pre-commit run --all-files
pytest -v
```

## CI/CD Testing

Tests run automatically on:
- Pull requests to main branch
- Manual workflow dispatch
- Release creation

### GitHub Actions Workflow

The CI runs:
1. Code quality checks (black, isort, ruff, mypy)
2. Full test suite with coverage
3. Coverage upload to Codecov

### Optimizing CI Minutes

To minimize GitHub Actions usage:
- Fast tests run on every PR
- Slow/integration tests can be marked with `@pytest.mark.slow`
- Docker builds only on releases or manual dispatch
- Coverage reports only uploaded once per test run

## Test Data

Example data is located in the main repository at `../../data/` (or can be referenced from `data/examples/` for backward compatibility):
- `MIC.csv`: Minimum Inhibitory Concentration data
- `AMR_genes.csv`: Antimicrobial resistance gene profiles
- Additional CSV files for comprehensive testing

The test fixtures in `tests/conftest.py` automatically copy data from the main repository's `data/` directory.

### Using Custom Test Data

```python
import pytest
from pathlib import Path

@pytest.fixture
def custom_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Add your test data here
    return data_dir
```

## Debugging Failed Tests

### Verbose Output
```bash
pytest -vv -s
```

### Show Print Statements
```bash
pytest -s
```

### Run Specific Test
```bash
pytest tests/test_workflow.py::test_data_loading_workflow -v
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

### Show Full Traceback
```bash
pytest --tb=long
```

## Writing New Tests

### Test File Naming
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Example Test

```python
import pytest
from strepsuis_mdr.analyzer import MDRAnalyzer
from strepsuis_mdr.config import Config

@pytest.mark.integration
def test_my_workflow(tmp_path):
    """Test description."""
    # Setup
    config = Config(
        data_dir="data/examples",
        output_dir=str(tmp_path / "output")
    )
    
    # Execute
    analyzer = MDRAnalyzer(config=config)
    
    # Assert
    assert analyzer.config is not None
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit  # Fast unit test
@pytest.mark.integration  # Integration test with real data
@pytest.mark.slow  # Long-running test
@pytest.mark.local  # Local development only
```

## Performance Considerations

### Test Speed Optimization

1. **Use minimal test data**: Reduce dataset size for faster tests
2. **Reduce bootstrap iterations**: Use 100 instead of 500+ for tests
3. **Mock external dependencies**: Avoid network calls or heavy computations
4. **Parallel testing**: Use pytest-xdist for faster execution

```bash
# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

## Continuous Integration

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This runs:
- black (formatting)
- isort (import sorting)
- ruff (linting)
- mypy (type checking)
- bandit (security checks)

### Local CI Simulation

Simulate the full CI pipeline locally:

```bash
# Code quality
black --check --line-length=100 .
isort --check --profile=black --line-length=100 .
ruff check .
mypy --ignore-missing-imports --no-strict-optional .

# Tests
pytest --cov --cov-report=xml --cov-report=term
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "FileNotFoundError"
- **Solution**: Ensure example data exists in the main repository's `../../data/` directory. The test fixtures automatically copy data from there.

**Issue**: Coverage too low
- **Solution**: Add workflow tests or integration tests that exercise real code paths

**Issue**: Tests timeout
- **Solution**: Mark slow tests with `@pytest.mark.slow` and reduce test data size

**Issue**: Import errors
- **Solution**: Install package in editable mode: `pip install -e .[dev]`

## Coverage Goals

### Priority Areas for Testing

1. **High Priority** (>90% coverage):
   - Core analysis functions
   - Data validation
   - Configuration handling
   - Error handling

2. **Medium Priority** (>70% coverage):
   - Utility functions
   - Reporting functions
   - CLI interface

3. **Lower Priority** (>50% coverage):
   - Visualization code
   - Interactive components

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Codecov](https://codecov.io/)

## Getting Help

If you encounter issues with testing:
1. Check this guide
2. Review existing tests in `tests/`
3. Check CI logs in GitHub Actions
4. Open an issue on GitHub

## Contributing

When contributing:
1. Write tests for new features
2. Maintain or improve coverage
3. Run full test suite before submitting PR
4. Update documentation if adding new test categories

## Test Data Documentation

### Input Data Format

All tests use CSV files with a `Strain_ID` column (or equivalent identifier):
- **Binary data**: 0 = absence, 1 = presence
- **MIC data**: Ordinal resistance levels or binary (0/1)
- **Required files**: `MIC.csv`, `AMR_genes.csv`
- **Optional files**: `Virulence.csv`, `MLST.csv`, `Serotype.csv`, etc.

### Example Data Details

Located in `examples/` directory:
```
examples/
├── MIC.csv              # 92 strains × 13 antibiotics
├── AMR_genes.csv        # 92 strains × gene profiles
├── Virulence.csv        # 92 strains × virulence factors
├── MLST.csv             # Sequence typing data
├── Serotype.csv         # Serological classification
├── MGE.csv              # Mobile genetic elements
├── Plasmid.csv          # Plasmid presence/absence
└── README.md            # Data format documentation
```

### Reference Outputs

Reference outputs for validation are generated during local testing:
- HTML reports with expected structure and content
- Excel workbooks with required sheets and data
- Statistical results with expected ranges

To generate reference outputs locally:
```bash
# Run full pipeline with example data
pytest tests/test_end_to_end.py::test_full_pipeline_with_validation -v
```

## Coverage Improvement Roadmap

### Current Status (Baseline)
- **Total coverage**: ~37% (before enhancement)
- **Core modules**: ~11-17% (analyzer, core)
- **Test modules**: 100%
- **Config modules**: 88-100%

### Target Coverage Goals

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `analyzer.py` | 17% | 80% | HIGH |
| `mdr_analysis_core.py` | 11% | 70% | HIGH |
| `cli.py` | 0% | 70% | MEDIUM |
| `excel_report_utils.py` | 0% | 60% | MEDIUM |
| `config.py` | 88% | 95% | LOW |

### Improvement Strategy

1. **Phase 1: Core Analysis (Weeks 1-2)**
   - Add end-to-end workflow tests ✅
   - Mock interactive components for testing
   - Validate data preprocessing pipeline
   - **Target**: 50% total coverage

2. **Phase 2: Output Validation (Weeks 3-4)**
   - Add reference output validation
   - Test report generation
   - Validate statistical computations
   - **Target**: 65% total coverage

3. **Phase 3: Edge Cases (Weeks 5-6)**
   - Test error handling
   - Boundary condition testing
   - Performance testing
   - **Target**: 80% total coverage

### How to Contribute to Coverage

1. **Identify uncovered code**:
   ```bash
   pytest --cov --cov-report=html
   open htmlcov/index.html
   # Click on module names to see uncovered lines
   ```

2. **Write targeted tests**:
   - Focus on red/yellow highlighted code
   - Test one function at a time
   - Include edge cases

3. **Verify improvement**:
   ```bash
   pytest --cov --cov-report=term-missing
   # Check coverage % increased
   ```

## Local vs CI Testing Strategy

### CI Testing (GitHub Actions)
**Goal**: Fast feedback, < 5 minutes per run

```bash
# Runs on every PR
pytest -m "not slow" --cov --cov-report=xml
```

**Characteristics**:
- Mini datasets (10 strains)
- Reduced bootstrap iterations (100)
- Fast integration tests only
- Coverage reporting to Codecov

### Local Testing (Developer Machine)
**Goal**: Comprehensive validation before release

```bash
# Full test suite
pytest -v --cov --cov-report=html

# Slow/comprehensive tests
pytest -v

# Specific comprehensive test
pytest tests/test_end_to_end.py::test_full_pipeline_with_validation -v
```

**Characteristics**:
- Full datasets (92 strains)
- Full bootstrap iterations (500+)
- All tests including slow ones
- Detailed HTML coverage reports

## Automated Coverage Reports

### Generate Coverage Badge

```bash
cd /path/to/separated_repos
python generate_coverage_badge.py
```

This creates:
- `COVERAGE_SUMMARY.md` - Summary table with badges
- `coverage_report.json` - Machine-readable coverage data
- Badge URLs for README files

### Add Badge to README

Copy the generated badge markdown:
```markdown
[![Coverage](https://img.shields.io/badge/coverage-XX%25-color)](link-to-coverage)
```

## Best Practices for Test Development

### 1. Use Real Example Data
```python
@pytest.fixture
def example_data():
    """Load real example data for testing."""
    example_dir = Path(__file__).parent.parent / "examples"
    return {
        "mic": pd.read_csv(example_dir / "MIC.csv"),
        "amr": pd.read_csv(example_dir / "AMR_genes.csv")
    }
```

### 2. Create Mini Datasets for Speed
```python
@pytest.fixture
def mini_dataset(tmp_path, example_data):
    """Create mini dataset (10 strains) for fast testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Subset to first 10 rows
    for name, df in example_data.items():
        df.head(10).to_csv(data_dir / f"{name}.csv", index=False)
    
    return data_dir
```

### 3. Validate Output Structure
```python
def test_output_structure(results):
    """Validate generated files have expected structure."""
    assert "html_reports" in results
    assert len(results["html_reports"]) > 0
    
    for html_path in results["html_reports"]:
        content = Path(html_path).read_text()
        assert len(content) > 1000
        assert "<!DOCTYPE html>" in content
```

### 4. Test Reproducibility
```python
def test_reproducibility(config):
    """Ensure analyses are reproducible."""
    analyzer1 = MDRAnalyzer(config=config)
    results1 = analyzer1.run()
    
    analyzer2 = MDRAnalyzer(config=config)
    results2 = analyzer2.run()
    
    assert results1["total_files"] == results2["total_files"]
```

## Performance Benchmarks

### Expected Test Execution Times

| Test Category | CI Time | Local Time |
|---------------|---------|------------|
| Unit tests | 1-2s | 1-2s |
| Integration (fast) | 3-5s | 3-5s |
| Workflow tests | 5-10s | 10-30s |
| End-to-end (mini) | 5-10s | 10-20s |
| End-to-end (full) | Skipped | 30-60s |
| **Total (CI)** | **< 5 min** | N/A |
| **Total (Local)** | N/A | **< 10 min** |

### GitHub Actions Budget

With current strategy:
- **Tests per PR**: 3-5 minutes
- **PRs per month**: ~10
- **Total monthly usage**: 30-50 minutes
- **Free tier limit**: 2,000 minutes/month
- **Usage**: ~2-3% of free tier ✅

## Future Enhancements

### Planned Improvements
1. [ ] Add integration tests for all analysis methods
2. [ ] Create reference output validation suite
3. [ ] Add performance regression tests
4. [ ] Implement mutation testing for critical paths
5. [ ] Add property-based testing for data validation
6. [ ] Create visual regression tests for plots
7. [ ] Add stress tests with large datasets (1000+ strains)

### Coverage Targets by Module
- [ ] `analyzer.py`: 80% → 90%
- [ ] `mdr_analysis_core.py`: 70% → 85%
- [ ] `cli.py`: 70% → 85%
- [ ] `excel_report_utils.py`: 60% → 75%
- [ ] Overall: 80% → 90%
