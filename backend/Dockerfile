# Backend — FastAPI + ML serving image (CPU only)
FROM python:3.12-slim

# Non-root user for production hygiene.
RUN groupadd --system app && useradd --system --gid app app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BASE_DIR=/app

WORKDIR /app

# Dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code + shared `src` package.
COPY src ./src
COPY backend ./backend

# Trained model artifacts (produce with `python scripts/train.py` first).
COPY models ./models

# Non-root runtime.
RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
