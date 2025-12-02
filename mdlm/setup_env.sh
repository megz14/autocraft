#!/bin/bash
# Complete environment setup script for MDLM
# This script creates the conda environment and installs flash-attn separately

set -e

ENV_NAME="mdlm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up MDLM environment..."
echo "=============================="
echo ""

# Step 1: Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Step 2: Create environment from requirements.yaml
echo ""
echo "Creating conda environment from requirements.yaml..."
conda env create -f "${SCRIPT_DIR}/requirements.yaml"

# Step 3: Activate environment
echo ""
echo "Activating environment: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Step 4: Install flash-attn separately (requires PyTorch to be installed first)
echo ""
echo "Installing flash-attn (this may take 10-30 minutes)..."
if [ -f "${SCRIPT_DIR}/install_flash_attn.sh" ]; then
    bash "${SCRIPT_DIR}/install_flash_attn.sh"
else
    echo "Warning: install_flash_attn.sh not found. Installing flash-attn manually..."
    export CUDA_HOME=$CONDA_PREFIX
    pip install flash-attn==2.5.6 --no-build-isolation
fi

# Step 5: Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python -c "import flash_attn; print('✓ Flash attention installed')" || echo "⚠ Flash attention installation failed - you may need to install it manually"
python -c "import transformers; print('✓ Transformers installed')"

echo ""
echo "=============================="
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""


