#!/usr/bin/env bash

set -e  # Exit on error

echo "Setting up autocraft environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create conda environment from requirements.yaml
echo "Creating conda environment 'mdlm'..."
cd mdlm
conda env create -f requirements.yaml

# Activate the environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate mdlm

# Install PyTorch with CUDA 12.4 support
echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn 
echo "Installing flash-attn..."
pip install flash-attn || echo "Warning: flash-attn installation failed (this is optional)"

# Install point-e
echo "Installing point-e..."
cd ../point-e
pip install -e .

cd ..

echo "Setup complete! Activate the environment with: conda activate mdlm"

