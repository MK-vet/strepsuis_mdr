# Security Policy

## Supported Versions

The following versions of StrepSuisMDR are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. Do Not Create Public Issues

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report Privately

Send a detailed report to the maintainers by:
- Creating a private security advisory on GitHub
- Or contacting the repository maintainers directly

### 3. Include Details

When reporting, please include:
- Type of vulnerability (e.g., injection, authentication, data exposure)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if available)
- Impact of the issue, including how an attacker might exploit it
- Python version and operating system
- StrepSuisMDR version affected

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Triage**: Within 5 business days
- **Resolution**: Depending on severity, typically within 30 days

## Security Best Practices

When using StrepSuisMDR:

### Data Handling
- Do not include sensitive patient identifiers in input data
- Use anonymized strain IDs
- Store output reports securely

### Dependencies
- Keep all dependencies up to date
- Use `pip-audit` or similar tools to check for known vulnerabilities
- Pin dependency versions in production environments

### Environment
- Run analyses in isolated environments (containers, virtual environments)
- Limit file system access to required directories only
- Do not run with elevated privileges unless necessary

## Known Security Considerations

### Input Validation
- All input files are validated before processing
- File paths are sanitized to prevent directory traversal
- Maximum file sizes are configurable

### Output Files
- Output reports may contain summary statistics from input data
- Review outputs before sharing to ensure no sensitive information is exposed

### External Connections
- The software does not make external network connections during analysis
- Report generation may load external JavaScript libraries (e.g., Plotly CDN) if HTML reports are opened in a browser

## Dependency Security

We regularly audit our dependencies for security vulnerabilities:

```bash
# Check for known vulnerabilities
pip-audit

# Check for outdated packages
pip list --outdated
```

Dependencies are pinned with upper bounds to prevent breaking changes while maintaining security updates.

---

**Last Updated**: 2025-01-15
