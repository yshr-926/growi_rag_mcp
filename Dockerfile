# Multi-stage build for optimal image size and security
FROM ghcr.io/astral-sh/uv:0.4.19 AS uv

# Production runtime stage
FROM python:3.11-slim AS runtime

# Security: Create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies and cleanup in single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy uv from builder stage
COPY --from=uv /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and install dependencies (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source code
COPY src/ ./src/
COPY config.yaml ./

# Copy models for local inference (as per spec requirement)
COPY models/ ./models/

# Create chroma_db directory with proper permissions
RUN mkdir -p /app/chroma_db && \
    chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Health check for container orchestration (MCP is STDIO-based)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pgrep -f "python.*src/main.py" || exit 1

# Expose MCP server port
EXPOSE 3000

# Start the MCP server
CMD ["python", "src/main.py"]