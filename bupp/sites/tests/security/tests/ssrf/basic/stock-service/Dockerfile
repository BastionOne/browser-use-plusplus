FROM python:3.11-slim

WORKDIR /app

# Install dependencies (no httpx needed here)
RUN pip install --no-cache-dir fastapi uvicorn

COPY app.py .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
