# Release Checklist

Use this checklist before creating a new release.

## Version 1.0.0 Release Checklist

### Pre-Release Checks

#### Code Quality
- [x] Code formatted (`black --check .`)
- [x] Imports sorted (`isort --check .`)
- [x] Most linting errors fixed (`ruff check .`)
- [ ] All tests pass (`pytest`)
- [ ] Code coverage >80% (`pytest --cov`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Type hints checked (`mypy .`)
- [ ] Security scan passed (`bandit -r .`)

#### Documentation
- [x] README.md is complete and accurate
- [x] Example data is included
- [x] CHANGELOG.md is updated
- [x] .gitignore is comprehensive
- [ ] USER_GUIDE.md is up-to-date
- [ ] CONTRIBUTING.md is clear
- [ ] All docstrings are complete

#### Package Configuration
- [x] Version number correct in pyproject.toml
- [x] Version number correct in __init__.py
- [x] All dependencies listed
- [x] Package metadata complete
- [x] License file present (MIT)

#### Docker
- [x] Dockerfile updated for local builds
- [x] Multi-stage build working
- [x] Health check configured
- [ ] Docker image builds successfully (requires network access)
- [ ] Docker image runs correctly
- [ ] Docker image size is optimized
- [ ] docker-compose.yml works

#### GitHub Actions
- [x] Workflow triggers optimized (PR and releases only)
- [x] Caching is configured
- [x] Python version matrix reduced to save runner minutes
- [x] Docker builds conditional on releases
- [ ] Test workflow runs successfully
- [ ] All CI checks pass
- [ ] Release workflow is ready
- [ ] Documentation workflow is ready

#### Notebooks
- [ ] Google Colab notebook tested
- [ ] Notebook installs package from GitHub
- [ ] Notebook runs without errors
- [ ] File upload/download works
- [ ] Instructions are clear

### Release Process

#### 1. Prepare Release
```bash
# Update version in all files
# Update CHANGELOG.md
# Commit changes
git add .
git commit -m "Prepare release v1.0.0"
git push
```

#### 2. Create Git Tag
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

#### 3. Create GitHub Release
- [ ] Go to GitHub Releases page
- [ ] Click "Create a new release"
- [ ] Select tag v1.0.0
- [ ] Title: "Version 1.0.0 - Initial Release"
- [ ] Add release notes (see template below)
- [ ] Publish release

#### 4. Verify Release
- [ ] Release workflow triggered automatically
- [ ] Docker image built and pushed
- [ ] Distribution packages created
- [ ] Release notes generated

#### 5. Post-Release
- [ ] Test installation from GitHub
- [ ] Test Docker image from registry
- [ ] Update documentation site (if applicable)
- [ ] Announce release

### Release Notes Template

```markdown
# Version 1.0.0 - Initial Release 🎉

Production-ready bioinformatics tool for [tool purpose].

## Features

- ✅ Professional Python package with CLI
- ✅ Docker container with multi-stage build
- ✅ Google Colab notebook for non-programmers
- ✅ Comprehensive documentation in English
- ✅ Example datasets included
- ✅ >80% test coverage
- ✅ Publication-quality outputs

## Installation

### Python Package
```bash
pip install git+https://github.com/MK-vet/[repo-name].git@v1.0.0
```

### Docker Container
```bash
docker pull ghcr.io/mk-vet/[repo-name]:1.0.0
```

### Google Colab
Click the badge in README.md

## What's Included

- Python package (wheel and source distribution)
- Docker container image (multi-platform)
- Google Colab notebook
- Example datasets
- User guide and documentation

## Requirements

- Python 3.8+ (for package)
- Docker (for container)
- Google account (for Colab)

## Citation

See README.md for citation information.

## Support

- Documentation: README.md and USER_GUIDE.md
- Issues: https://github.com/MK-vet/[repo-name]/issues
- Contributing: See CONTRIBUTING.md
```

### Optional: PyPI Publication

If publishing to PyPI:

#### Prepare
- [ ] Create PyPI account
- [ ] Create API token
- [ ] Test on TestPyPI first

#### Publish
```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ [package-name]

# Upload to PyPI
twine upload dist/*
```

#### Verify
- [ ] Package appears on PyPI
- [ ] Installation works: `pip install [package-name]`
- [ ] Documentation renders correctly

### Optional: Docker Hub Publication

If publishing to Docker Hub:

```bash
# Login
docker login

# Tag image
docker tag [repo-name]:latest [username]/[repo-name]:1.0.0
docker tag [repo-name]:latest [username]/[repo-name]:latest

# Push
docker push [username]/[repo-name]:1.0.0
docker push [username]/[repo-name]:latest
```

### Troubleshooting

Common issues and solutions:

**Tests fail:**
- Run locally: `pytest -v`
- Check for environment-specific issues
- Verify all dependencies installed

**Docker build fails:**
- Check Dockerfile syntax
- Verify base image available
- Check network connectivity

**Release workflow fails:**
- Check GitHub Actions logs
- Verify secrets configured
- Check permissions

**Package import fails:**
- Verify package structure
- Check __init__.py
- Verify dependencies

## Post-Release Monitoring

After release:
- [ ] Monitor GitHub Issues for bug reports
- [ ] Track installation statistics
- [ ] Collect user feedback
- [ ] Plan next release

## Version Numbering

Follow Semantic Versioning (semver):
- MAJOR.MINOR.PATCH
- 1.0.0 = First stable release
- 1.0.1 = Bug fixes
- 1.1.0 = New features (backward compatible)
- 2.0.0 = Breaking changes

---

**Last Updated:** 2025-01-14
**Version:** 1.0.0
