# Flash Attention Installation Guide

`flash-attn` requires CUDA development headers to compile. If you encounter compilation errors during installation, follow these steps:

## Solution 1: Use conda environment with CUDA toolkit (Recommended)

The `requirements.yaml` has been updated to include `cuda-toolkit`. Recreate your conda environment:

```bash
conda env remove -n mdlm  # Remove existing environment if needed
conda env create -f requirements.yaml
conda activate mdlm
```

Then install flash-attn:
```bash
pip install flash-attn==2.5.6 --no-build-isolation
```

## Solution 2: Set CUDA_HOME environment variable

If Solution 1 doesn't work, manually set CUDA_HOME before installing:

1. Find your CUDA installation:
   ```bash
   conda activate mdlm
   which nvcc  # This will show the path to nvcc
   ```

2. Set CUDA_HOME (usually the parent directory of bin/nvcc):
   ```bash
   export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
   # Or if CUDA is installed in a specific location:
   # export CUDA_HOME=/usr/local/cuda-12.4  # Adjust version as needed
   ```

3. Install flash-attn:
   ```bash
   pip install flash-attn==2.5.6 --no-build-isolation
   ```

## Solution 3: Install CUDA toolkit via conda

If CUDA headers are still missing, install the full CUDA toolkit:

```bash
conda activate mdlm
conda install -c nvidia cuda-toolkit=12.4
export CUDA_HOME=$CONDA_PREFIX
pip install flash-attn==2.5.6 --no-build-isolation
```

## Solution 4: Use pre-built wheels (if available)

Check if pre-built wheels are available for your CUDA/Python version:
```bash
pip install flash-attn==2.5.6 --no-build-isolation
# If this fails, you'll need to build from source using one of the solutions above
```

## Troubleshooting

- **Error: `cuda_runtime_api.h: No such file or directory`**: CUDA headers are missing. Use Solution 1 or 3.
- **CUDA version mismatch warnings**: Usually safe to ignore if versions are close (e.g., 12.1 vs 12.4).
- **Build takes a long time**: This is normal - flash-attn compilation can take 10-30 minutes.

## Verify Installation

After installation, verify it works:
```python
python -c "import flash_attn; print('Flash attention installed successfully!')"
```


