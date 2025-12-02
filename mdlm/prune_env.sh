#!/bin/bash
# Script to prune/clean the mdlm conda environment

set -e

ENV_NAME="${CONDA_DEFAULT_ENV:-mdlm}"

echo "Pruning conda environment: $ENV_NAME"
echo "================================"

# Check if environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Activating conda environment: $ENV_NAME"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
fi

echo ""
echo "1. Cleaning conda package cache..."
conda clean --packages --tarballs --yes

echo ""
echo "2. Cleaning pip cache..."
pip cache purge 2>/dev/null || echo "   (pip cache already clean)"

echo ""
echo "3. Checking environment size..."
ENV_SIZE=$(du -sh $CONDA_PREFIX 2>/dev/null | cut -f1)
echo "   Current size: $ENV_SIZE"

echo ""
echo "4. Listing installed packages..."
echo "   Conda packages: $(conda list | wc -l) packages"
echo "   Pip packages: $(pip list | wc -l) packages"

echo ""
echo "Pruning complete!"
echo ""
echo "To do a deeper clean, you can:"
echo "  - Remove the environment and recreate: conda env remove -n $ENV_NAME && conda env create -f requirements.yaml"
echo "  - Clean index cache: conda clean --index-cache --yes"
echo ""


