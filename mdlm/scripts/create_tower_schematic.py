#!/usr/bin/env python3
"""Script to create a tower schematic file."""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import create_tower_schematic


def main():
    parser = argparse.ArgumentParser(description="Create a tower schematic file")
    parser.add_argument(
        "--output",
        type=str,
        default="tower_schematic.npy",
        help="Output path for the tower schematic (default: tower_schematic.npy)"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Size of the voxel grid (default: 32)"
    )
    parser.add_argument(
        "--center_x",
        type=int,
        default=16,
        help="X coordinate of tower center (default: 16)"
    )
    parser.add_argument(
        "--center_z",
        type=int,
        default=16,
        help="Z coordinate of tower center (default: 16)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="Height of the tower (default: 20)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="Width of the tower (radius from center, default: 3)"
    )
    parser.add_argument(
        "--block_id",
        type=int,
        default=1,
        help="Block ID to use for the tower (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"Creating tower schematic:")
    print(f"  Block size: {args.block_size}")
    print(f"  Center: ({args.center_x}, {args.center_z})")
    print(f"  Height: {args.height}")
    print(f"  Width: {args.width}")
    print(f"  Block ID: {args.block_id}")
    print()
    
    # Create tower schematic
    tower = create_tower_schematic(
        block_size=args.block_size,
        center_x=args.center_x,
        center_z=args.center_z,
        height=args.height,
        width=args.width,
        block_id=args.block_id
    )
    
    # Count occupied blocks
    block_ids = tower[..., 0]
    num_occupied = (block_ids > 0).sum()
    
    print(f"Created tower with {num_occupied} occupied blocks")
    print(f"Saving to: {args.output}")
    
    # Save schematic
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tower)
    
    print(f"✓ Saved tower schematic to {output_path}")
    print(f"  Shape: {tower.shape}")
    print(f"  Occupied blocks: {num_occupied}")


if __name__ == "__main__":
    main()

