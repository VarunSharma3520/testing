FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (safe in Docker)
RUN pip install --no-cache-dir uv

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8000

# Run your app
CMD ["uv", "run", "uvicorn", "utils.app:app", "--host", "0.0.0.0", "--port", "8000"]
