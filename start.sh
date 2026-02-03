#!/bin/bash
set -e

echo "Starting CCTV Missing Person Pipeline"

cd /workspace/Face-Recognition-System

# Pull latest code (GitHub integration)
git pull origin main || true

# Create required directories
mkdir -p videos reference/face reference/body outputs/videos outputs/logs

# Run pipeline
python main.py
