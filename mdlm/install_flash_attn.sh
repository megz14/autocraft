#!/bin/bash
# Helper script to install flash-attn with proper CUDA setup

set -e

echo "Installing flash-attn with CUDA support..."

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Please activate your conda environment first:"
    echo "  conda activate mdlm"
    exit 1
fi

# Try to find CUDA_HOME automatically
if [ -z "$CUDA_HOME" ]; then
    # Check if nvcc is available
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
        echo "Found CUDA at: $CUDA_HOME"
        export CUDA_HOME
    elif [ -d "$CONDA_PREFIX" ]; then
        # Check conda environment for CUDA
        if [ -d "$CONDA_PREFIX/include/cuda" ] || [ -d "$CONDA_PREFIX/include" ]; then
            CUDA_HOME=$CONDA_PREFIX
            echo "Using conda environment CUDA: $CUDA_HOME"
            export CUDA_HOME
        fi
    fi
fi

# Verify CUDA headers exist
if [ -n "$CUDA_HOME" ]; then
    if [ ! -f "$CUDA_HOME/include/cuda_runtime_api.h" ] && [ ! -f "$CUDA_HOME/include/cuda/cuda_runtime_api.h" ]; then
        echo "Warning: CUDA headers not found at $CUDA_HOME/include/"
        echo "Attempting to install cuda-toolkit via conda..."
        conda install -y -c nvidia cuda-toolkit || conda install -y -c conda-forge cudatoolkit-dev
        if [ -d "$CONDA_PREFIX" ]; then
            export CUDA_HOME=$CONDA_PREFIX
        fi
    fi
else
    echo "CUDA_HOME not set. Attempting to install cuda-toolkit via conda..."
    conda install -y -c nvidia cuda-toolkit || conda install -y -c conda-forge cudatoolkit-dev
    if [ -d "$CONDA_PREFIX" ]; then
        export CUDA_HOME=$CONDA_PREFIX
    fi
fi

echo "Installing flash-attn==2.5.6..."
pip install flash-attn==2.5.6 --no-build-isolation

echo "Verifying installation..."
python -c "import flash_attn; print('✓ Flash attention installed successfully!')" || {
    echo "✗ Installation verification failed"
    exit 1
}

echo "Done!"


