# Building Generator Quick Start Guide

## Step 1: Create a Building Schematic

Use the `--type` parameter to select which building to create. Currently supports:
- `tower` - A solid vertical tower
- `well` - A hollow circular well structure

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

### Customize Your Building

**Tower parameters:**
- `--height`: How tall (default: 20)
- `--width`: Radius/base size (default: 3)

**Well parameters:**
- `--height`: How tall (default: 15)
- `--outer_radius`: Outer wall radius (default: 5)
- `--inner_radius`: Inner hollow radius (default: 3)

**Common parameters (for both):**
- `--center_x`: X position center (default: 16)
- `--center_z`: Z position center (default: 16)
- `--block_id`: Block type ID (default: 1, doesn't matter for generation)

## Step 2: Use the Schematic for Sampling

After creating a schematic, use it with the `eval.coordinate_file` parameter:

```bash
python main.py \
  mode=sample_eval \
  data=craft3d \
  data.craft3d_dir=dataset/3dcraft_normalized \
  model=small \
  backbone=dit \
  parameterization=subs \
  eval.checkpoint_path=/path/to/checkpoint.ckpt \
  eval.coordinate_file=tower_schematic.npy \
  sampling.steps=128 \
  sampling.num_sample_batches=1 \
  loader.eval_batch_size=1 \
  +wandb.offline=true
```

The model will generate new block IDs for the coordinates in your schematic!

