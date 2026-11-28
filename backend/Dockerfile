FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for psycopg2 and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY src/ /app/src/
COPY app/ /app/app/
COPY backend/ /app/backend/
COPY database/ /app/database/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY outputs/ /app/outputs/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
