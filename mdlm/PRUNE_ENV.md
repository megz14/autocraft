# Conda Environment Pruning Guide

This guide shows different ways to prune/clean your conda environment.

## Option 1: Remove Unused Packages (Recommended)

Remove packages that are not explicitly listed in your environment file:

```bash
conda activate mdlm

# List all packages
conda list

# Remove unused packages (keeps dependencies of packages you need)
conda clean --all

# Or remove specific unused packages manually
conda remove <package-name>
```

## Option 2: Clean Conda Cache

Remove cached package files and temporary files:

```bash
# Clean package cache (frees up disk space)
conda clean --packages

# Clean tarballs
conda clean --tarballs

# Clean index cache
conda clean --index-cache

# Clean all: packages, tarballs, and index cache
conda clean --all

# Clean with confirmation prompt
conda clean --all --yes
```

## Option 3: Recreate Environment from Scratch (Cleanest)

This is the most thorough way to prune - completely recreate the environment:

```bash
# 1. Export current environment (optional - for backup)
conda env export > env_backup.yaml

# 2. Remove the environment
conda deactivate  # if currently active
conda env remove -n mdlm

# 3. Recreate from requirements.yaml
conda env create -f requirements.yaml

# 4. Activate the new environment
conda activate mdlm

# 5. Install flash-attn separately if needed
bash install_flash_attn.sh
```

## Option 4: Remove Specific Packages

Remove packages you don't need:

```bash
conda activate mdlm

# Remove a specific package
conda remove <package-name>

# Remove multiple packages
conda remove package1 package2 package3

# Remove with dependencies check
conda remove <package-name> --force  # Forces removal even if dependencies break
```

## Option 5: Prune Based on Current Requirements

Update your environment to match exactly what's in requirements.yaml:

```bash
conda activate mdlm

# Export current environment
conda env export > current_env.yaml

# Compare with requirements.yaml and manually remove extras
# Or recreate from requirements.yaml (see Option 3)
```

## Option 6: Minimal Cleanup (Just Pip)

If you just want to clean pip packages:

```bash
conda activate mdlm

# List pip packages
pip list

# Remove specific pip package
pip uninstall <package-name>

# Remove all pip packages (then reinstall from requirements.txt)
pip freeze > pip_freeze.txt
pip uninstall -r pip_freeze.txt -y
pip install -r requirements.txt
```

## Option 7: Prune System-Wide Conda

Clean conda system-wide (affects all environments):

```bash
# Clean all conda caches
conda clean --all

# Clean with size report
conda clean --all --dry-run

# Clean and show what will be removed
conda clean --all --verbose
```

## Quick Prune Script

Here's a quick script to prune the current environment:

```bash
#!/bin/bash
# Quick prune script for mdlm environment

conda activate mdlm

echo "Cleaning conda cache..."
conda clean --all --yes

echo "Removing unused pip packages..."
pip cache purge

echo "Environment size check:"
du -sh $CONDA_PREFIX

echo "Done pruning!"
```

## Check Environment Size

Before and after pruning, check the size:

```bash
conda activate mdlm
du -sh $CONDA_PREFIX
# Or
du -sh $(conda info --base)/envs/mdlm
```

## Verify Environment Integrity

After pruning, verify everything still works:

```bash
conda activate mdlm

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print('Flash attention OK')"
python -c "import transformers; print('Transformers OK')"

# Or run a quick test
python -c "import sys; print('Python:', sys.version)"
```


