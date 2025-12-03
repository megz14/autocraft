#!/usr/bin/env python3
"""
All-in-one pipeline: Text → Point Cloud → Voxel Grid → Block IDs → Schematic.npy

This script:
1. Generates a point cloud from text using point-e
2. Downsamples to 32x32x32 voxel grid using majority vote
3. Samples block IDs using the trained Craft3D diffusion model
4. Saves final schematic.npy file

Usage:
    python text_to_schematic.py \
        --text "a red motorcycle" \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --output schematic.npy \
        --model small \
        --steps 128
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict

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
import dataloader
import diffusion
from main import _blocks_to_schematic


def point_cloud_to_voxel_coords(point_cloud, block_size=32, bounds=None):
    """Convert point cloud to occupied voxel coordinates using majority vote.
    
    Args:
        point_cloud: Point cloud object from point-e, or numpy array of shape (N, 3) with (x, y, z) coords
        block_size: Size of voxel grid (default: 32)
        bounds: Optional bounds tuple ((min_x, min_y, min_z), (max_x, max_y, max_z))
                If None, auto-calculates from point cloud
    
    Returns:
        numpy array of shape (N, 3) with (x, y, z) coordinates centered at (0, 0, 0)
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
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
    else:
        min_coords, max_coords = bounds
    
    # Center and scale to [0, block_size)
    coord_range = max_coords - min_coords
    coord_range = np.where(coord_range < 1e-8, 1.0, coord_range)  # Avoid division by zero
    coords_normalized = (coords - min_coords) / coord_range * block_size
    coords_normalized = np.clip(coords_normalized, 0, block_size - 1)
    
    # Use set to get unique voxel coordinates (majority vote handled by uniqueness)
    voxel_coords_set = set()
    for i in range(len(coords_normalized)):
        x, y, z = coords_normalized[i]
        # Convert to voxel indices
        x_idx = int(np.floor(x))
        y_idx = int(np.floor(y))
        z_idx = int(np.floor(z))
        
        # Bounds check
        if 0 <= x_idx < block_size and 0 <= y_idx < block_size and 0 <= z_idx < block_size:
            voxel_coords_set.add((x_idx, y_idx, z_idx))
    
    # Convert to numpy array and center at (0, 0, 0)
    offset = block_size // 2
    voxel_coords = np.array(list(voxel_coords_set), dtype=np.float32)
    
    if len(voxel_coords) > 0:
        # Center coordinates
        voxel_coords = voxel_coords - offset
    else:
        voxel_coords = np.zeros((0, 3), dtype=np.float32)
    
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
    # Register resolvers (same as in mdlm/main.py)
    omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
    omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    omegaconf.OmegaConf.register_new_resolver('eval', eval)
    omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    
    # Resolve config path relative to script directory
    if not os.path.isabs(config_path):
        config_path = str(Path(__file__).parent / config_path)
    
    # Load config
    with hydra.initialize(config_path=config_path, version_base=None):
        overrides = [
            'data=craft3d',
            f'eval.checkpoint_path={checkpoint_path}',
        ]
        if config_overrides:
            overrides.extend(config_overrides)
        
        config = hydra.compose(config_name='config', overrides=overrides)
    
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    
    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
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
    """All-in-one pipeline: Text → Point Cloud → Voxel Grid → Block IDs → Schematic.
    
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
    voxel_coords = point_cloud_to_voxel_coords(point_cloud, block_size=block_size)
    print(f"✓ Extracted {len(voxel_coords)} occupied voxels")
    
    if len(voxel_coords) == 0:
        raise ValueError("No occupied voxels found in point cloud!")
    
    # Step 4: Load diffusion model and sample block IDs
    print(f"\nStep 4/5: Loading diffusion model and sampling block IDs...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model: {model_name}")
    print(f"  Sampling steps: {sampling_steps}")
    
    config_overrides = [
        f'model={model_name}',
        f'sampling.steps={sampling_steps}',
    ]
    
    diffusion_model, config, tokenizer = load_diffusion_model(
        checkpoint_path,
        config_path=config_path,
        config_overrides=config_overrides
    )
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    
    # Prepare coordinates for diffusion model
    seq_len = config.model.length
    num_coords = len(voxel_coords)
    
    # Pad coordinates to seq_len
    coords_tensor = torch.zeros((1, seq_len, 3), dtype=torch.float32, device=device)
    coords_tensor[0, :num_coords] = torch.from_numpy(voxel_coords).to(device)
    
    # Create attention mask
    attention_mask = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    attention_mask[0, :num_coords] = 1
    
    print(f"  Prepared coordinates: {num_coords} occupied voxels (seq_len={seq_len})")
    
    # Sample block IDs
    with torch.no_grad():
        block_ids = diffusion_model.restore_model_and_sample(
            num_steps=sampling_steps,
            coords=coords_tensor[0]  # Shape: (seq_len, 3)
        )
    
    # Extract only non-padding positions
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    valid_mask = attention_mask[0].cpu().bool()
    valid_block_ids = block_ids[0][valid_mask].cpu().numpy()  # Shape: (num_coords,)
    valid_coords = voxel_coords  # Already filtered
    
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
        description='All-in-one pipeline: Text → Point Cloud → Voxel Grid → Block IDs → Schematic.npy'
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

