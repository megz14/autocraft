#!/usr/bin/env python3
"""Script to create building schematic files (tower, well, etc.)."""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import create_tower_schematic, create_well_schematic


def main():
    parser = argparse.ArgumentParser(description="Create building schematic files")
    parser.add_argument(
        "--type",
        type=str,
        choices=["tower", "well"],
        default="tower",
        help="Type of building to create (default: tower)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the schematic (default: <type>_schematic.npy)"
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
        help="X coordinate of building center (default: 16)"
    )
    parser.add_argument(
        "--center_z",
        type=int,
        default=16,
        help="Z coordinate of building center (default: 16)"
    )
    
    # Tower-specific arguments
    parser.add_argument(
        "--height",
        type=int,
        default=20,
        help="Height of the building (default: 20)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="Width/radius of the building (default: 3, for tower: radius from center)"
    )
    
    # Well-specific arguments
    parser.add_argument(
        "--outer_radius",
        type=int,
        default=5,
        help="Outer radius for well (default: 5)"
    )
    parser.add_argument(
        "--inner_radius",
        type=int,
        default=3,
        help="Inner radius for well - hollow space (default: 3)"
    )
    
    parser.add_argument(
        "--block_id",
        type=int,
        default=1,
        help="Block ID to use for the building (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        args.output = f"{args.type}_schematic.npy"
    
    print(f"Creating {args.type} schematic:")
    print(f"  Block size: {args.block_size}")
    print(f"  Center: ({args.center_x}, {args.center_z})")
    print(f"  Block ID: {args.block_id}")
    print()
    
    # Create schematic based on type
    if args.type == "tower":
        print(f"  Height: {args.height}")
        print(f"  Width (radius): {args.width}")
        print()
        schematic = create_tower_schematic(
            block_size=args.block_size,
            center_x=args.center_x,
            center_z=args.center_z,
            height=args.height,
            width=args.width,
            block_id=args.block_id
        )
    elif args.type == "well":
        print(f"  Height: {args.height}")
        print(f"  Outer radius: {args.outer_radius}")
        print(f"  Inner radius: {args.inner_radius}")
        print()
        schematic = create_well_schematic(
            block_size=args.block_size,
            center_x=args.center_x,
            center_z=args.center_z,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
            height=args.height,
            block_id=args.block_id
        )
    else:
        raise ValueError(f"Unknown building type: {args.type}")
    
    # Count occupied blocks
    block_ids = schematic[..., 0]
    num_occupied = (block_ids > 0).sum()
    
    print(f"Created {args.type} with {num_occupied} occupied blocks")
    print(f"Saving to: {args.output}")
    
    # Save schematic
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, schematic)
    
    print(f"✓ Saved {args.type} schematic to {output_path}")
    print(f"  Shape: {schematic.shape}")
    print(f"  Occupied blocks: {num_occupied}")


if __name__ == "__main__":
    main()

