#!/usr/bin/env python3
"""Script to normalize Craft3D dataset schematics."""

import sys
from pathlib import Path
import numpy as np
import json

# Add parent directory to path to import helper module
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper.craft3d_dataset import Craft3DDataset


def normalize_schematic(schematic_path: Path, output_path: Path, block_size: int = 32):
    """Normalize a single schematic file.
    
    Args:
        schematic_path: Path to the schematic.npy file
        output_path: Path to save the normalized schematic
        block_size: Target size for normalization
    """
    print(f"Loading schematic from: {schematic_path}")
    schematic = np.load(schematic_path)
    print(f"  Original shape: {schematic.shape}")
    print(f"  Original dtype: {schematic.dtype}")
    
    # Use the normalization method from Craft3DDataset
    # Create a minimal instance to access the method
    temp_dataset = Craft3DDataset.__new__(Craft3DDataset)
    normalized = temp_dataset._resize_schematic(schematic, block_size=block_size)
    
    if normalized is None:
        print(f"  Warning: Schematic is empty, skipping...")
        return False
    
    print(f"  Normalized shape: {normalized.shape}")
    print(f"  Normalized dtype: {normalized.dtype}")
    
    # Save normalized schematic
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized)
    print(f"  Saved to: {output_path}")
    
    return True


def normalize_dataset(data_dir: str, block_size: int = 32, output_dir: str = None):
    """Normalize all schematics in the dataset.
    
    Args:
        data_dir: Directory containing the houses
        block_size: Target size for normalization
        output_dir: Optional output directory (defaults to data_dir with _normalized suffix)
    """
    data_path = Path(data_dir)
    houses_dir = data_path / "houses"
    
    if not houses_dir.exists():
        print(f"Error: Houses directory not found at {houses_dir}")
        return
    
    if output_dir is None:
        output_dir = str(data_path) + "_normalized"
    output_path = Path(output_dir)
    output_houses_dir = output_path / "houses"
    output_houses_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy splits.json if it exists
    splits_file = data_path / "splits.json"
    if splits_file.exists():
        import shutil
        shutil.copy(splits_file, output_path / "splits.json")
        print(f"Copied splits.json to {output_path / 'splits.json'}")
    
    # Find all house directories
    house_dirs = [d for d in houses_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(house_dirs)} house directories")
    print(f"Output directory: {output_path}")
    print(f"Block size: {block_size}\n")
    
    success_count = 0
    fail_count = 0
    
    for house_dir in house_dirs:
        schematic_path = house_dir / "schematic.npy"
        
        if not schematic_path.exists():
            print(f"Skipping {house_dir.name}: schematic.npy not found")
            continue
        
        output_house_dir = output_houses_dir / house_dir.name
        output_house_dir.mkdir(parents=True, exist_ok=True)
        output_schematic_path = output_house_dir / "schematic.npy"
        
        print(f"\nProcessing: {house_dir.name}")
        if normalize_schematic(schematic_path, output_schematic_path, block_size):
            # Copy other files if they exist
            for other_file in ["placed.json", "block_counts.json", "stats.json"]:
                src = house_dir / other_file
                if src.exists():
                    import shutil
                    shutil.copy(src, output_house_dir / other_file)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Normalization complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize Craft3D dataset schematics")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/3dcraft",
        help="Path to the Craft3D data directory"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Target block size for normalization (default: 32)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir + '_normalized')"
    )
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir
    
    normalize_dataset(str(data_dir), block_size=args.block_size, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

