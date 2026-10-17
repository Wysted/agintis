# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# uv extras to install: LLM providers (openai, anthropic, ollama, deepseek), `otel` (tracing, needed when
# OBSERVABILITY_TRACING=otlp) and `ocr` (parser ocr_index_card; its models ship in the rapidocr wheel,
# so OCR works offline), space or comma separated; empty = none.
# Default "openai otel"; example: docker build --build-arg UV_EXTRAS="anthropic otel ocr" .
ARG UV_EXTRAS="openai otel"

# Dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    set -eu; extras=""; \
    for e in $(echo "$UV_EXTRAS" | tr ',' ' '); do extras="$extras --extra $e"; done; \
    uv sync --frozen --no-dev --no-editable --no-install-project $extras

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; extras=""; \
    for e in $(echo "$UV_EXTRAS" | tr ',' ' '); do extras="$extras --extra $e"; done; \
    uv sync --frozen --no-dev --no-editable $extras

# Tokenizer baked into the image so the app boots offline (no HF hub access at runtime).
# Pinned: BAAI/bge-m3 at a fixed commit, verified by checksum (the LFS sha256 published by the hub).
# To change the embedding model, bump the repo/commit/checksum and EMBEDDINGS_TOKENIZER.
FROM scratch AS tokenizer
ADD --checksum=sha256:21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08 \
    https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json \
    /bge-m3/tokenizer.json

FROM python:3.12.15-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH" \
    EMBEDDINGS_TOKENIZER=/app/models/bge-m3/tokenizer.json
RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=tokenizer --chown=app:app /bge-m3 /app/models/bge-m3
# Alembic (run on demand: `docker compose --profile app run --rm migrate`).
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
USER app
EXPOSE 8000
CMD ["uvicorn", "agintis.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
