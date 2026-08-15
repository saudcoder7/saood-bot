# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a wheelhouse
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app.py faqs.json ./
COPY public/ ./public/

# Pre-download NLTK data at build time so cold starts are fast
RUN python -c "\
import nltk; \
pkgs = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']; \
[nltk.download(p, quiet=True) for p in pkgs]"

EXPOSE 8080

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]
