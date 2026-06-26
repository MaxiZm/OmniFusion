FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

# Fix (medium Dockerfile): Copy lockfile BEFORE syncing so uv can use --locked.
# This ensures reproducible builds using the exact pinned dependency versions.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Fix (medium Dockerfile): COPY . . now respects .dockerignore so .env, .git,
# .venv, and local data/ are NOT baked into the image layers.
COPY . .
RUN uv sync --locked --no-dev

# Non-root user
RUN useradd -m omni && chown -R omni:omni /app
USER omni

# Create data directory for the database
RUN mkdir -p /app/data

# Pin to 1 worker as per invariant #1
CMD ["uv", "run", "uvicorn", "omnifusion.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
