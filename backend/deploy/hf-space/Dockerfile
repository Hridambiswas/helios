# backend/deploy/hf-space/Dockerfile
#
# Hugging Face Docker Space build. The Space repo will be laid out with
# this Dockerfile at the root and the backend/* Python files copied in
# via the export script (Part C of 004). See README.md for the Space
# YAML front matter (sdk: docker, app_port: 7860).
#
# Notable HF Space constraints (docs/hub/en/spaces-sdks-docker):
#   - Container runs as UID 1000 → we create the user explicitly.
#   - Only inbound port 7860 is exposed → nginx is not used here; uvicorn
#     listens on 0.0.0.0:7860 behind HF's own TLS-terminating proxy.
#   - COPY --chown=user avoids the recursive chown that would double the
#     image layer size.
#   - /data volume is a paid feature — we run entirely on ephemeral
#     rootfs (Redis in-memory, Chroma under /tmp).
#
# Outbound port constraint on Spaces means Postgres MUST go through
# Supabase (port 5432 is on the HF allow-list — see 003_report.md §Hard
# constraint). MinIO and OTel are turned off via env vars.

FROM python:3.11-slim

# Set up a non-root user with the HF-required UID before any COPY.
RUN useradd -m -u 1000 user

# System deps: gcc/g++ for asyncpg + fastembed, curl for the HEALTHCHECK,
# redis-server so start.sh can bring up an in-process Redis alongside
# uvicorn (Redis is required by rate-limit + brute-force middleware and
# HF Spaces block outbound port 6379).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl redis-server \
    && rm -rf /var/lib/apt/lists/*

# Home + PATH for the user; app under $HOME/app.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    APP_HOME=/home/user/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HF Space defaults
    APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    # Embedded Chroma under /tmp — ephemeral, but that's the Space
    # storage model. Persistent state lives in Supabase.
    CHROMA_PATH=/tmp/chroma \
    # MinIO / Celery / OTel disabled — see 003_report.md §3(c).
    MINIO_ENABLED=false \
    OTEL_EXPORTER_OTLP_ENDPOINT= \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    CELERY_BROKER_URL=redis://127.0.0.1:6379/1 \
    CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/2 \
    OAUTH_FRONTEND_URL=https://helios-hridam.vercel.app \
    OAUTH_BACKEND_URL=https://hridam-helios.hf.space \
    VERIFIER_ENABLED=true \
    GROQ_MODEL=openai/gpt-oss-120b \
    GEMINI_MODEL=gemini-2.5-flash \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

WORKDIR $APP_HOME

# Requirements first for layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-warm the embedding model into ~/.cache/fastembed at build time so
# the Space cold-start does not have to download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Rest of the app.
COPY --chown=user . .
RUN chmod +x start.sh

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

# start.sh brings up redis + uvicorn in the same container.
CMD ["./start.sh"]
