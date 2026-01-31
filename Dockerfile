# Multi-stage build for StrepSuisMDR
# Stage 1: Builder - Install dependencies and package
FROM python:3.11-slim as builder

LABEL maintainer="MK-vet <support@strepsuis-suite.org>"
LABEL description="StrepSuisMDR: Python Workflow for Integrated Phenotype–Genotype Analysis of Multidrug Resistance in Streptococcus suis"
LABEL version="1.0.0"

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the package source code
WORKDIR /build
COPY . /build/

# Install the package from local source
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: Runtime - Minimal image with only runtime dependencies
FROM python:3.11-slim

LABEL maintainer="MK-vet <support@strepsuis-suite.org>"
LABEL description="StrepSuisMDR: Python Workflow for Integrated Phenotype–Genotype Analysis of Multidrug Resistance in Streptococcus suis"
LABEL version="1.0.0"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create directories for data and output
RUN mkdir -p /data /output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV OUTPUT_DIR=/output
ENV PATH="/opt/venv/bin:$PATH"

# Add non-root user for security
RUN useradd -m -u 1000 biouser && \
    chown -R biouser:biouser /app /data /output

# Switch to non-root user
USER biouser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD strepsuis-mdr --version || exit 1

# Set the entrypoint to the CLI command
ENTRYPOINT ["strepsuis-mdr"]

# Default command shows help
CMD ["--help"]
