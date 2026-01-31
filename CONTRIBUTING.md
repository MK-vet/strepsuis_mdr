# Contributing to StrepSuisMDR

Thank you for your interest in contributing to StrepSuisMDR! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in communication
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/strepsuis-mdr.git
   cd strepsuis-mdr
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/MK-vet/strepsuis-mdr.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### Installation for Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. These will run automatically before each commit:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Fast linting
- **mypy**: Type checking
- **bandit**: Security checks

Run manually on all files:
```bash
pre-commit run --all-files
```

## Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our style guidelines

3. **Test your changes** thoroughly (see Testing section)

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

## Testing

### Running Tests

Run the full test suite:
```bash
pytest
```

Run tests with coverage report:
```bash
pytest --cov --cov-report=term-missing
```

Run specific test file:
```bash
pytest tests/test_specific.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Aim for >80% code coverage
- Include both unit tests and integration tests

Example test:
```python
def test_analyzer_initialization():
    """Test that analyzer initializes correctly."""
    analyzer = Analyzer(data_dir="./data")
    assert analyzer.data_dir == "./data"
```

## Submitting Changes

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes

3. **PR Requirements**:
   - All tests must pass
   - Code coverage should not decrease
   - Pre-commit hooks should pass
   - Include clear description of changes
   - Reference any related issues

## Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (enforced by black)
- **Imports**: Sorted with isort (black-compatible profile)
- **Type hints**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings

Example:
```python
def analyze_data(
    data_dir: str,
    output_dir: str,
    bootstrap_iterations: int = 500
) -> dict:
    """Analyze bacterial genomic data.
    
    Args:
        data_dir: Directory containing input CSV files
        output_dir: Directory for output files
        bootstrap_iterations: Number of bootstrap iterations (default: 500)
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        FileNotFoundError: If data_dir does not exist
        ValueError: If bootstrap_iterations < 1
    """
    # Implementation
    pass
```

### Documentation Style

- Use **Markdown** for documentation files
- Include **code examples** where helpful
- Keep documentation **up-to-date** with code changes
- Write in **clear, simple English**

### Commit Message Guidelines

Good commit messages help maintain project history:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Provide detailed description in message body if needed

Examples:
```
Add bootstrap confidence interval calculation

Update README with new installation instructions

Fix issue with missing data handling
- Check for NaN values before analysis
- Add appropriate error message
- Include test for edge case
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages or logs

### Feature Requests

When suggesting features:

- Describe the problem it solves
- Provide use case examples
- Consider implementation approaches
- Discuss potential impact on existing features

## Questions and Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For sensitive issues or private inquiries

## Recognition

Contributors will be recognized in:

- CHANGELOG.md for significant contributions
- GitHub contributors page
- Release notes for version-specific contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Additional Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Git Workflow Guide](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

## Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!

---

**Note**: This project is under active development. Guidelines may evolve. Check this document regularly for updates.
