#!/usr/bin/env python3
"""

This script:
1. Generates a point cloud from text using point-e
2. Downsamples to 32x32x32 voxel grid
3. Samples block IDs using the trained diffusion model
4. Saves final schematic.npy file

Usage:
    python text_to_schematic.py \
        --text "a house" \
        --checkpoint_path /workspace/autocraft/autocraft/mdlm/outputs/craft3d_train/2025.12.03/000628/checkpoints/best.ckpt \
        --output schematic.npy \
        --model small \
        --steps 128
    
    Or with relative path:
    python text_to_schematic.py \
        --text "a house" \
        --checkpoint_path mdlm/outputs/craft3d_train/2025.12.03/000628/checkpoints/best.ckpt \
        --output schematic.npy
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from scipy.ndimage import binary_closing, generate_binary_structure

# Add parent directory to path for point-e imports
autocraft_dir = Path(__file__).parent
sys.path.insert(0, str(autocraft_dir))

# Add mdlm to path for MDLM imports
sys.path.insert(0, str(autocraft_dir / 'mdlm'))

# Point-e imports
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

# MDLM imports
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

# Register resolvers before importing mdlm modules (they might already register them)
def _register_resolvers_safe():
    """Register OmegaConf resolvers, skipping if already registered."""
    resolvers = {
        'cwd': os.getcwd,
        'device_count': torch.cuda.device_count,
        'eval': eval,
        'div_up': lambda x, y: (x + y - 1) // y,
    }
    for name, func in resolvers.items():
        try:
            omegaconf.OmegaConf.register_new_resolver(name, func)
        except ValueError:
            pass  # Already registered

_register_resolvers_safe()

import dataloader
import diffusion

# Copy _blocks_to_schematic function to avoid importing main.py (which has resolver registration)
def _blocks_to_schematic(block_ids, coords, attention_mask=None, pad_token_id=0, block_size=32):
    """Convert block IDs and coordinates to a voxel grid schematic.
    
    Coordinates are centered at (0,0,0) and mapped to [0, block_size) range.
    Matches normalized schematic.npy format: (y, z, x, 2) where channel 0 is block IDs.
    
    Args:
        block_ids: Tensor of shape (seq_len,) containing block IDs
        coords: Tensor of shape (seq_len, 3) containing (x, y, z) coordinates (centered at 0,0,0)
        attention_mask: Optional tensor of shape (seq_len,) to filter padding
        pad_token_id: Padding token ID to filter out
        block_size: Size of the voxel grid (default: 32)
        
    Returns:
        numpy array of shape (block_size, block_size, block_size, 2) matching schematic.npy format
        Format: (y, z, x, 2) where channel 0 is block IDs, channel 1 is metadata (set to 0)
    """
    # Filter out padding positions
    if attention_mask is not None:
        valid_mask = (attention_mask == 1)
        block_ids = block_ids[valid_mask]
        coords = coords[valid_mask]
    else:
        # Filter out padding tokens
        valid_mask = (block_ids != pad_token_id)
        block_ids = block_ids[valid_mask]
        coords = coords[valid_mask]
    
    # Convert to numpy
    if isinstance(block_ids, torch.Tensor):
        block_ids = block_ids.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    
    # Create empty voxel grid (y, z, x, 2) format like normalized schematic.npy
    schematic = np.zeros((block_size, block_size, block_size, 2), dtype=np.uint8)
    
    # Determine if coordinates need shifting based on their range
    # Dataset normalization keeps coordinates in [0, block_size) range (positive)
    # So coordinates should already be in the correct range
    needs_shift = False
    if len(coords) > 0:
        min_coord = float(coords.min())
        max_coord = float(coords.max())
        # If coordinates are in [0, block_size) range, use directly (matches dataset format)
        if min_coord >= 0 and max_coord < block_size:
            needs_shift = False
            offset = 0
        # If we see negative coordinates, they're centered at 0 and need shifting to [0, block_size)
        elif min_coord < 0:
            needs_shift = True
            offset = block_size // 2
        else:
            # Coordinates are positive but might be out of range - clip to [0, block_size)
            needs_shift = False
            offset = 0
            # Will be clipped in bounds check below
    else:
        offset = 0
    
    # Place blocks in the grid
    placed_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    
    for i in range(len(block_ids)):
        block_id = int(block_ids[i])
        x, y, z = coords[i]  # Coordinates are in (x, y, z) format
        
        # Apply offset if needed
        if needs_shift:
            x = x + offset
            y = y + offset
            z = z + offset
        
        # Round to integers
        x = int(np.round(x))
        y = int(np.round(y))
        z = int(np.round(z))
        
        # Check bounds before clipping
        x_valid = 0 <= x < block_size
        y_valid = 0 <= y < block_size
        z_valid = 0 <= z < block_size
        
        if not (x_valid and y_valid and z_valid):
            out_of_bounds_count += 1
            continue
        
        # Skip if block_id is padding (0)
        if block_id == pad_token_id or block_id == 0:
            continue
        
        # Place block in schematic (format is y, z, x, channels)
        # Schematic storage: schematic[y, z, x] where y=height, z=depth, x=width
        # Channel 0: block ID
        # Channel 1: metadata (set to 0 for now since we don't have that info)
        if schematic[y, z, x, 0] > 0:
            collision_count += 1
        
        schematic[y, z, x, 0] = block_id
        schematic[y, z, x, 1] = 0
        placed_count += 1
    
    # Debug: print statistics if there are issues
    if len(block_ids) > 0:
        if out_of_bounds_count > 0 or collision_count > 0 or placed_count < len(block_ids) * 0.9:
            print(f'[Schematic conversion] Placed {placed_count}/{len(block_ids)} blocks, '
                  f'{out_of_bounds_count} out-of-bounds, {collision_count} collisions, '
                  f'coords range: [{min_coord:.1f}, {max_coord:.1f}], needs_shift: {needs_shift}')
    
    return schematic


def point_cloud_to_voxel_coords(
    point_cloud, 
    block_size=32, 
    bounds=None,
):
    """Convert point cloud to occupied voxel coordinates using morphological downsampling.
    
    Uses morphological operations to fill gaps and smooth surfaces.
    
    Args:
        point_cloud: Point cloud object from point-e, or numpy array of shape (N, 3) with (x, y, z) coords
        block_size: Size of voxel grid (default: 32)
        bounds: Optional bounds tuple ((min_x, min_y, min_z), (max_x, max_y, max_z))
                If None, auto-calculates from point cloud
    
    Returns:
        numpy array of shape (N, 3) with (y, z, x) coordinates in [0, block_size) range
        Format matches Minecraft/schematic format: (y, z, x) where y is height
    """
    # Extract coordinates from point cloud
    if hasattr(point_cloud, 'coords'):
        coords = point_cloud.coords  # Shape: (N, 3)
    elif isinstance(point_cloud, np.ndarray):
        coords = point_cloud
    else:
        raise ValueError(f"Unsupported point cloud format: {type(point_cloud)}")
    
    # Normalize coordinates to [0, block_size) range
    if bounds is None:
        min_coords = coords.min(axis=0)  # Minimum for each axis (x, y, z)
        max_coords = coords.max(axis=0)  # Maximum for each axis (x, y, z)
    else:
        min_coords, max_coords = bounds
    
    # Shift coordinates to be positive by subtracting minimum value from each axis
    # This ensures all coordinates become positive (matching dataset format)
    # coords_shifted will have all values >= 0
    coords_shifted = coords - min_coords
    
    # Scale to fit within block_size
    coord_range = max_coords - min_coords
    coord_range = np.where(coord_range < 1e-8, 1.0, coord_range)  # Avoid division by zero
    coords_normalized = coords_shifted / coord_range * block_size
    coords_normalized = np.clip(coords_normalized, 0, block_size - 1)
    
    # Verify all coordinates are now positive (should be [0, block_size))
    assert np.all(coords_normalized >= 0), f"Some coordinates are still negative after shifting! Min: {coords_normalized.min()}"
    
    # Apply morphological downsampling
    # Start with simple voxelization
    voxel_grid = np.zeros((block_size, block_size, block_size), dtype=bool)
    for i in range(len(coords_normalized)):
        x, y, z = coords_normalized[i]
        x_idx = int(np.floor(x))
        y_idx = int(np.floor(y))
        z_idx = int(np.floor(z))
        if 0 <= x_idx < block_size and 0 <= y_idx < block_size and 0 <= z_idx < block_size:
            voxel_grid[y_idx, z_idx, x_idx] = True
    
    # Apply morphological closing to fill small gaps (creates more blocky structure)
    structure = generate_binary_structure(3, 1)  # 6-connected
    voxel_grid = binary_closing(voxel_grid, structure=structure, iterations=1)
    print(f"  Morphological voxelization: {voxel_grid.sum()} occupied voxels from {len(coords_normalized)} points")
    
    # Get coordinates of occupied voxels
    occupied_positions = np.argwhere(voxel_grid)  # Shape: (N, 3) with indices (y, z, x)
    
    if len(occupied_positions) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Keep in (y, z, x) format to match Minecraft/schematic format
    # Note: Dataset normalization keeps coordinates in [0, block_size) range (not centered at 0)
    # So we keep coordinates in [0, block_size) range to match training data format
    # These will be converted to (x, y, z) before passing to the model
    voxel_coords = occupied_positions.astype(np.float32)  # Keep (y, z, x) format, in [0, block_size) range
    
    # Verify coordinates are in [0, block_size) range (they should be since occupied_positions are indices)
    if len(voxel_coords) > 0:
        assert np.all(voxel_coords >= 0) and np.all(voxel_coords < block_size), \
            f"Coordinates should be in [0, {block_size}) range, but got range [{voxel_coords.min():.1f}, {voxel_coords.max():.1f}]"
    
    return voxel_coords


def initialize_point_e_sampler(device, base_name='base40M-textvec'):
    """Initialize point-e sampler for text-to-point-cloud generation."""
    print(f'Creating point-e base model ({base_name})...')
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    
    print('Creating point-e upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    
    print('Downloading point-e checkpoints...')
    base_model.load_state_dict(load_checkpoint(base_name, device))
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
    )
    
    return sampler


def load_diffusion_model(checkpoint_path, config_path='mdlm/configs', config_overrides=None):
    """Load the Craft3D diffusion model from checkpoint."""
    # Note: Resolvers are already registered when importing from main.py
    
    # Get absolute path to config directory
    script_dir = Path(__file__).parent
    config_dir = script_dir / config_path
    
    # Ensure config directory exists
    if not config_dir.exists():
        raise ValueError(f"Config directory does not exist: {config_dir}")
    
    # Convert to absolute path
    config_dir_abs = str(config_dir.resolve())
    
    # Load config using initialize_config_dir (accepts absolute paths)
    with hydra.initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        overrides = [
            'data=craft3d',
            f'eval.checkpoint_path="{checkpoint_path}"',  # Quote path to handle special characters
        ]
        if config_overrides:
            overrides.extend(config_overrides)
        
        config = hydra.compose(config_name='config', overrides=overrides)
    
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Resolve checkpoint path to absolute
    if os.path.isabs(checkpoint_path):
        checkpoint_path_abs = str(Path(checkpoint_path).resolve())
    else:
        # Resolve relative to script directory
        checkpoint_path_abs = str(Path(script_dir) / checkpoint_path)
        checkpoint_path_abs = str(Path(checkpoint_path_abs).resolve())
    
    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path_abs,
        tokenizer=tokenizer,
        config=config
    )
    
    return model, config, tokenizer


def text_to_schematic_pipeline(
    text_prompt,
    checkpoint_path,
    output_path,
    block_size=32,
    model_name='small',
    sampling_steps=128,
    device=None,
    config_path='mdlm/configs',
    point_e_base='base40M-textvec',
):
    """Text → Point Cloud → Voxel Grid → Block IDs → Schematic.
    
    Args:
        text_prompt: Text description to generate from
        checkpoint_path: Path to Craft3D diffusion model checkpoint
        output_path: Path to save final schematic.npy
        block_size: Voxel grid size (default: 32)
        model_name: Model size name (default: 'small')
        sampling_steps: Number of diffusion sampling steps (default: 128)
        device: Device to run on (default: auto-detect)
        config_path: Path to configs directory (default: 'mdlm/configs')
        point_e_base: Point-e base model name (default: 'base40M-textvec')
    
    Returns:
        numpy array of shape (block_size, block_size, block_size, 2) - the final schematic
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Text to Schematic Pipeline")
    print("=" * 70)
    
    # Step 1: Initialize point-e sampler
    print(f"\nStep 1/5: Initializing point-e models...")
    point_e_sampler = initialize_point_e_sampler(device, base_name=point_e_base)
    
    # Step 2: Generate point cloud from text
    print(f"\nStep 2/5: Generating point cloud from text: '{text_prompt}'")
    samples = None
    for x in tqdm(
        point_e_sampler.sample_batch_progressive(
            batch_size=1, 
            model_kwargs=dict(texts=[text_prompt])
        ), 
        desc="Point cloud generation"
    ):
        samples = x
    
    point_cloud = point_e_sampler.output_to_point_clouds(samples)[0]
    print(f"✓ Generated point cloud with {len(point_cloud.coords)} points")
    
    # Step 3: Convert to voxel coordinates
    print(f"\nStep 3/5: Converting point cloud to {block_size}x{block_size}x{block_size} voxel grid...")
    voxel_coords = point_cloud_to_voxel_coords(
        point_cloud, 
        block_size=block_size,
    )
    print(f"✓ Extracted {len(voxel_coords)} occupied voxels")
    
    if len(voxel_coords) == 0:
        raise ValueError("No occupied voxels found in point cloud!")
    
    # Step 4: Load diffusion model and sample block IDs
    print(f"\nStep 4/5: Loading diffusion model and sampling block IDs...")
    # Resolve checkpoint path to absolute for display
    checkpoint_path_abs = str(Path(checkpoint_path).resolve())
    print(f"  Checkpoint: {checkpoint_path_abs}")
    print(f"  Model: {model_name}")
    print(f"  Sampling steps: {sampling_steps}")
    
    config_overrides = [
        f'model={model_name}',
        f'sampling.steps={sampling_steps}',
        'loader.eval_batch_size=1',  # We only have one sequence to generate
    ]
    
    diffusion_model, config, tokenizer = load_diffusion_model(
        checkpoint_path_abs,
        config_path=config_path,
        config_overrides=config_overrides
    )
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    
    # Prepare coordinates for diffusion model
    seq_len = config.model.length
    num_coords = len(voxel_coords)
    
    # Truncate coordinates if we have more than the model can handle
    if num_coords > seq_len:
        print(f"  Warning: {num_coords} occupied voxels exceeds model sequence length ({seq_len})")
        print(f"  Truncating to first {seq_len} voxels")
        voxel_coords = voxel_coords[:seq_len]
        num_coords = seq_len
    
    # Convert coordinates from (y, z, x) to (x, y, z) for the model
    # voxel_coords is in (y, z, x) format in [0, block_size) range, but model expects (x, y, z)
    # Format conversion: (y, z, x) -> (x, y, z)
    # Index mapping: [0=y, 1=z, 2=x] -> [0=x, 1=y, 2=z]
    voxel_coords_yzx = voxel_coords.copy()  # Keep original (y, z, x) for final schematic
    voxel_coords_xyz = np.zeros_like(voxel_coords_yzx)
    voxel_coords_xyz[:, 0] = voxel_coords_yzx[:, 2]  # x = x (from y,z,x index 2)
    voxel_coords_xyz[:, 1] = voxel_coords_yzx[:, 0]  # y = y (from y,z,x index 0)
    voxel_coords_xyz[:, 2] = voxel_coords_yzx[:, 1]  # z = z (keep positive, no flipping)
    
    # Ensure all coordinates are in [0, block_size) range (should already be, but verify)
    print(f"  Coordinate range before model: X=[{voxel_coords_xyz[:, 0].min():.1f}, {voxel_coords_xyz[:, 0].max():.1f}], "
          f"Y=[{voxel_coords_xyz[:, 1].min():.1f}, {voxel_coords_xyz[:, 1].max():.1f}], "
          f"Z=[{voxel_coords_xyz[:, 2].min():.1f}, {voxel_coords_xyz[:, 2].max():.1f}]")
    
    # Pad coordinates to seq_len
    coords_tensor = torch.zeros((1, seq_len, 3), dtype=torch.float32, device=device)
    coords_tensor[0, :num_coords] = torch.from_numpy(voxel_coords_xyz).to(device)
    
    # Create attention mask
    attention_mask = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    attention_mask[0, :num_coords] = 1
    
    print(f"  Prepared coordinates: {num_coords} occupied voxels (seq_len={seq_len})")
    print(f"  Converted from (y,z,x) to (x,y,z) format for model input")
    
    # Sample block IDs
    with torch.no_grad():
        block_ids = diffusion_model.restore_model_and_sample(
            num_steps=sampling_steps,
            coords=coords_tensor  # Shape: (batch, seq_len, 3) in (x, y, z) format
        )
    
    # Extract only non-padding positions
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    valid_mask = attention_mask[0].cpu().bool()
    valid_block_ids = block_ids[0][valid_mask].cpu().numpy()  # Shape: (num_coords,)
    # voxel_coords_xyz already contains only valid coordinates (no padding), so use it directly
    # It has shape (num_coords, 3) which matches valid_block_ids shape (num_coords,)
    valid_coords = voxel_coords_xyz  # Use (x, y, z) format - already contains only valid coordinates
    
    # Ensure coordinates are in [0, block_size) range (they should already be, but verify)
    print(f"  Coordinate range after sampling: X=[{valid_coords[:, 0].min():.1f}, {valid_coords[:, 0].max():.1f}], "
          f"Y=[{valid_coords[:, 1].min():.1f}, {valid_coords[:, 1].max():.1f}], "
          f"Z=[{valid_coords[:, 2].min():.1f}, {valid_coords[:, 2].max():.1f}]")
    
    # If coordinates are negative, shift them to be positive
    if valid_coords.min() < 0:
        print(f"  Warning: Some coordinates are negative! Shifting to positive range...")
        min_coords = valid_coords.min(axis=0)
        valid_coords = valid_coords - min_coords  # Shift to make all coordinates >= 0
        print(f"  After shifting: X=[{valid_coords[:, 0].min():.1f}, {valid_coords[:, 0].max():.1f}], "
              f"Y=[{valid_coords[:, 1].min():.1f}, {valid_coords[:, 1].max():.1f}], "
              f"Z=[{valid_coords[:, 2].min():.1f}, {valid_coords[:, 2].max():.1f}]")
    
    print(f"✓ Sampled {len(valid_block_ids)} block IDs")
    
    # Step 5: Create final schematic
    print(f"\nStep 5/5: Creating final schematic.npy...")
    schematic = _blocks_to_schematic(
        block_ids=torch.from_numpy(valid_block_ids),
        coords=torch.from_numpy(valid_coords),
        attention_mask=None,  # Already filtered
        pad_token_id=pad_token_id,
        block_size=block_size
    )
    
    # Save schematic
    output_path = Path(output_path)
    output_path = output_path.resolve()  # Make absolute path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, schematic)
    print(f"✓ Saved schematic to {output_path}")
    
    # Print statistics
    occupied_voxels = np.sum(schematic[..., 0] > 0)
    unique_blocks = len(np.unique(schematic[..., 0][schematic[..., 0] > 0]))
    print(f"\nSchematic statistics:")
    print(f"  Total voxels: {block_size**3}")
    print(f"  Occupied voxels: {occupied_voxels}")
    print(f"  Unique block types: {unique_blocks}")
    
    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)
    
    return schematic


def main():
    parser = argparse.ArgumentParser(
        description='Text → Point Cloud → Voxel Grid → Block IDs → Schematic.npy'
    )
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='Text prompt to generate from'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to Craft3D diffusion model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_schematic.npy',
        help='Output path for schematic.npy (default: output_schematic.npy)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='small',
        choices=['tiny', 'small', 'medium'],
        help='Model size (default: small)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=128,
        help='Number of diffusion sampling steps (default: 128)'
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=32,
        help='Voxel grid size (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='mdlm/configs',
        help='Path to configs directory (default: mdlm/configs)'
    )
    parser.add_argument(
        '--point_e_base',
        type=str,
        default='base40M-textvec',
        choices=['base40M-textvec', 'base300M-textvec', 'base1B-textvec'],
        help='Point-e base model name (default: base40M-textvec)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Run pipeline
    try:
        schematic = text_to_schematic_pipeline(
            text_prompt=args.text,
            checkpoint_path=args.checkpoint_path,
            output_path=args.output,
            block_size=args.block_size,
            model_name=args.model,
            sampling_steps=args.steps,
            device=device,
            config_path=args.config_path,
            point_e_base=args.point_e_base,
        )
        print(f"\nSuccess! Schematic saved to {args.output}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

