# Use official slim Python
FROM python:3.10-slim

# Create non-root user
ARG USER=appuser
ARG UID=1000
RUN adduser --disabled-password --gecos "" --uid ${UID} ${USER}

WORKDIR /app

# Install system deps (for scikit/lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY app /app/app
# Copy model(s)
COPY models /app/models

# Change to non-root user
USER ${USER}

# Expose port
EXPOSE 8080

# Use Gunicorn + Uvicorn workers for production robustness
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", \
     "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]