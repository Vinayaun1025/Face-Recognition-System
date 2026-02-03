# ===============================
# Base Image (CUDA + PyTorch)
# ===============================
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# ===============================
# Environment Variables
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cache models on volume (important for speed)
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# ===============================
# System Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Working Directory
# ===============================
WORKDIR /workspace/Face-Recognition-System

# ===============================
# Copy Only Requirements First
# (Docker layer caching)
# ===============================
COPY requirements.txt .

# ===============================
# Install Python Dependencies
# ===============================
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===============================
# Install PyTorch CUDA (RTX 3090)
# ===============================
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# ===============================
# Copy Project Code
# ===============================
COPY . .

# ===============================
# Permissions
# ===============================
RUN chmod +x start.sh

# ===============================
# Default Command
# ===============================
CMD ["bash", "start.sh"]
