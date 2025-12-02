# Building Schematic Generator Usage Guide

This guide explains how to create building schematics and use them for sampling.

## Creating Building Schematics

Use the `create_building_schematic.py` script to generate building schematics:

### Create a Tower

```bash
cd /Users/tyo/Coding/autocraft/autocraft/mdlm
python scripts/create_building_schematic.py \
  --type tower \
  --output tower_schematic.npy \
  --height 20 \
  --width 3
```

### Create a Well

```bash
python scripts/create_building_schematic.py \
  --type well \
  --output well_schematic.npy \
  --height 15 \
  --outer_radius 5 \
  --inner_radius 3
```

### Available Parameters

**Common parameters:**
- `--type`: Building type (`tower` or `well`)
- `--output`: Output file path (default: `<type>_schematic.npy`)
- `--block_size`: Voxel grid size (default: 32)
- `--center_x`: X coordinate of building center (default: 16)
- `--center_z`: Z coordinate of building center (default: 16)
- `--block_id`: Block ID to use (default: 1)

**Tower-specific:**
- `--height`: Height of the tower (default: 20)
- `--width`: Width/radius from center (default: 3)

**Well-specific:**
- `--height`: Height of the well structure (default: 15)
- `--outer_radius`: Outer radius of the well (default: 5)
- `--inner_radius`: Inner radius (hollow space, default: 3)

## Using Schematics for Sampling

Once you have a schematic file, use it for generation by specifying the `eval.coordinate_file` parameter:

```bash
python main.py \
  mode=sample_eval \
  data=craft3d \
  data.craft3d_dir=dataset/3dcraft_normalized \
  model=small \
  backbone=dit \
  parameterization=subs \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  eval.coordinate_file=/path/to/tower_schematic.npy \
  sampling.steps=128 \
  sampling.num_sample_batches=1 \
  loader.eval_batch_size=1 \
  +wandb.offline=true
```

### How It Works

1. The script loads coordinates from the schematic file
2. Extracts all occupied positions (where blocks exist)
3. Converts coordinates to centered format (at 0,0,0)
4. Uses those coordinates for sampling
5. The model generates new block IDs for those positions
6. Creates a new structure with the same shape but potentially different blocks

### Output

Generated schematics are saved in `schematics/` directory:
- `schematics/sample_0000_generated.npy` - Generated structure
- Both files have shape `(32, 32, 32, 2)` matching the original format

## Examples

### Example 1: Create and use a tower

```bash
# Create tower
python scripts/create_building_schematic.py \
  --type tower \
  --output simple-schematic/tower.npy \
  --height 20 \
  --width 3

# Generate samples using tower coordinates
python main.py \
  mode=sample_eval \
  data=craft3d \
  eval.coordinate_file=simple-schematic/tower.npy \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  sampling.steps=128
```

### Example 2: Create and use a well

```bash
# Create well
python scripts/create_building_schematic.py \
  --type well \
  --output simple-schematic/well.npy \
  --height 15 \
  --outer_radius 5 \
  --inner_radius 3

# Generate samples using well coordinates
python main.py \
  mode=sample_eval \
  data=craft3d \
  eval.coordinate_file=simple-schematic/well.npy \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  sampling.steps=128
```

