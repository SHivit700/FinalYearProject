FROM python:3.11-slim

WORKDIR /app

# System dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + torchvision together — must come from the same index
# to avoid the torchvision::nms version mismatch error
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

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

# Railway injects PORT at runtime; default to 8001 for local Docker runs
ENV PORT=8001

EXPOSE 8001

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
