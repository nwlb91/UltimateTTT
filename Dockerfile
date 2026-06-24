# Optional: containerized deploy (Fly.io, Railway, a VPS, etc.).
# Render uses render.yaml instead and does not need this file.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV UTTT_ORT_THREADS=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
