#!/bin/bash
# Entrypoint script for PTLFlow container
set -e

echo "========================================"
echo "PTLFlow Container"
echo "========================================"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# Check PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo "========================================"
echo "Container ready!"
echo "========================================"

# Execute the command passed to the container
exec "$@"
