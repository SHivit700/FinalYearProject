FROM python:3.11-slim

WORKDIR /app

# System dependencies for EasyOCR (opencv-python-headless needs no display libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + torchvision together — must come from the same index
# to avoid the torchvision::nms version mismatch error
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1
# Store EasyOCR models inside the image layer so they aren't re-downloaded on every cold start
ENV EASYOCR_MODULE_PATH=/app/.EasyOCR

# Install remaining dependencies (torch already satisfied, streamlit excluded — not needed for API)
COPY requirements.txt .
RUN grep -v "^streamlit\|^plotly" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Pre-download EasyOCR English models so the first request isn't slow
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application source
COPY src/ ./src/

# Create writable data directories (mount a Railway volume here to persist sessions)
RUN mkdir -p src/data/sessions src/data/uploads

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
