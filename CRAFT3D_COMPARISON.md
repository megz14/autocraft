# Comparison: User's Craft3DModule vs Existing Craft3D Loader

## Overview

The two implementations serve different purposes and work with different data formats, despite both using the Craft3D dataset.

## Key Differences

### 1. **Data Source & Format**

**User's Code (Craft3DModule):**
- Reads from `schematic.npy` files
- Each file contains a pre-computed 3D voxel grid: `(y, z, x, entryshape)`
- Static snapshot of the final house structure

**Existing Code (Craft3DDataset):**
- Reads from `placed.json` files
- Contains sequential block placement annotations with timestamps
- Tracks the building process step-by-step
- Format: `[timestamp, annotator_id, coordinate, block_info, action]`

### 2. **Data Representation**

**User's Code:**
- **Output**: Binary voxel grids of fixed size (32×32×32 by default)
- **Processing**: 
  - Extracts first channel (block_id)
  - Converts to binary (occupied/empty)
  - Centers around center of mass
  - Resizes/crops to `block_size`
  - Pads with zeros if smaller
- **Result**: `torch.FloatTensor` of shape `(block_size, block_size, block_size)`

**Existing Code:**
- **Output**: Complex dictionaries with multiple components
- **Processing**:
  - Converts sequential annotations to voxel representations
  - Creates local context (7×7×7) with block types
  - Creates global context (21×21×21) occupancy-only
  - Maintains history of previous steps
  - Generates targets for next block placements
- **Result**: 
  - Inputs: `{"local": (C*H, D, D, D), "global": (1, G, G, G), "center": (3,)}`
  - Targets: `{"coords": (A,), "types": (A,)}`

### 3. **Use Case & Model Type**

**User's Code:**
- Designed for **diffusion models** or **voxel-based generative models**
- Works with fixed-size voxel grids
- Suitable for unconditional generation or reconstruction tasks
- Each sample is a complete house structure

**Existing Code:**
- Designed for **autoregressive/sequential models** (like VoxelCNN)
- Predicts next block placement given building history
- Suitable for conditional generation tasks
- Each sample is a building step with context

### 4. **Framework Integration**

**User's Code:**
- Uses **PyTorch Lightning** (`LightningDataModule`)
- Integrates with Lightning's training loop
- Has `prepare_data()`, `setup()`, and dataloader methods
- Uses `save_hyperparameters()` for config management

**Existing Code:**
- Uses plain **PyTorch** (`torch.utils.data.Dataset`)
- More flexible, can be used with any framework
- Manual dataloader creation required

### 5. **Data Filtering**

**User's Code:**
- Filters out schematics with `sum() <= 0` (empty)
- Filters out schematics with `sum() <= 10` (too small)
- Uses splits from `splits.json` but has a bug: always uses `"train"` split in `_load_schemetics()`

**Existing Code:**
- Filters houses with `len(annotation) >= 100` (minimum block count)
- Filters valid items based on local distance constraints
- Properly uses the specified split (train/val/test)

### 6. **Coordinate System**

**User's Code:**
- Uses `(y, z, x)` ordering from numpy files
- Centers around center of mass
- Fixed coordinate system after centering

**Existing Code:**
- Uses `(x, y, z)` coordinates from JSON
- Centers around the last placed block
- Dynamic coordinate system that shifts with each step

### 7. **Batch Structure**

**User's Code:**
- Returns batches of shape `(batch_size, block_size, block_size, block_size)`
- Simple tensor format
- Single value per sample

**Existing Code:**
- Returns batches of dictionaries
- Multiple tensors per sample (local voxels, global voxels, center, targets)
- Complex structure for multi-task learning

### 8. **Bug in User's Code**

**Critical Issue:**
```python
def _load_schemetics(self, split):
    # ...
    for filename in self.splits["train"]:  # ❌ Always uses "train"!
        # Should be: self.splits[split]
```

This means validation and test sets will actually load training data.

## Summary Table

| Aspect | User's Craft3DModule | Existing Craft3DDataset |
|--------|---------------------|------------------------|
| **Data File** | `schematic.npy` | `placed.json` |
| **Format** | Voxel grid | Sequential annotations |
| **Output** | Binary voxel tensor | Dict with inputs/targets |
| **Size** | Fixed (32×32×32) | Variable (local 7×7×7, global 21×21×21) |
| **Centering** | Center of mass | Last block position |
| **Purpose** | Diffusion/generative models | Autoregressive models |
| **Framework** | PyTorch Lightning | Plain PyTorch |
| **History** | No | Yes (3 steps) |
| **Block Types** | Binary only | 256 block types |

## When to Use Which?

**Use User's Craft3DModule if:**
- Building a diffusion model for 3D house generation
- Need fixed-size voxel grids
- Working with PyTorch Lightning
- Want simple binary occupancy representation

**Use Existing Craft3DDataset if:**
- Building an autoregressive model (predict next block)
- Need sequential building context
- Want block type information (not just occupancy)
- Need local/global context windows
- Working with any PyTorch-based framework

