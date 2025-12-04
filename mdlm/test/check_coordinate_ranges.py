#!/usr/bin/env python3
"""Check all houses in dataset/3dcraft and print coordinate ranges."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import helper module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from helper.craft3d_dataset import Craft3DDataset


def check_coordinate_ranges(data_dir, subset='train', max_houses=None):
    """Check coordinate ranges for all houses in the dataset.
    
    Args:
        data_dir: Path to dataset directory (e.g., 'dataset/3dcraft')
        subset: Dataset subset to check ('train', 'val', or 'test')
        max_houses: Maximum number of houses to check (None for all)
    """
    print("=" * 80)
    print("COORDINATE RANGE CHECK")
    print("=" * 80)
    print(f"Dataset directory: {data_dir}")
    print(f"Subset: {subset}")
    print()
    
    # Load dataset
    try:
        dataset = Craft3DDataset(
            data_dir=data_dir,
            subset=subset,
            max_samples=max_houses
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    num_houses = dataset.get_num_houses()
    print(f"Total houses: {num_houses}")
    print()
    
    # Collect coordinate ranges
    all_x_min, all_x_max = [], []
    all_y_min, all_y_max = [], []
    all_z_min, all_z_max = [], []
    all_x_ranges, all_y_ranges, all_z_ranges = [], [], []
    
    print("Processing houses...")
    print("-" * 80)
    
    for i in range(num_houses):
        try:
            house = dataset.get_house(i)
            coords = house[:, 1:].cpu().numpy().astype(float)  # Extract x, y, z columns
            
            if len(coords) == 0:
                print(f"House {i:5d}: Empty (no blocks)")
                continue
            
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            all_x_min.append(x_min)
            all_x_max.append(x_max)
            all_y_min.append(y_min)
            all_y_max.append(y_max)
            all_z_min.append(z_min)
            all_z_max.append(z_max)
            all_x_ranges.append(x_range)
            all_y_ranges.append(y_range)
            all_z_ranges.append(z_range)
            
            # Print per-house info
            print(f"House {i:5d}: "
                  f"X=[{x_min:7.1f}, {x_max:7.1f}] (span: {x_range:6.1f}) | "
                  f"Y=[{y_min:7.1f}, {y_max:7.1f}] (span: {y_range:6.1f}) | "
                  f"Z=[{z_min:7.1f}, {z_max:7.1f}] (span: {z_range:6.1f}) | "
                  f"Blocks: {len(coords):5d}")
            
        except Exception as e:
            print(f"House {i:5d}: Error - {e}")
            continue
    
    print("-" * 80)
    print()
    
    # Overall statistics
    if len(all_x_min) == 0:
        print("No houses processed successfully!")
        return
    
    print("=" * 80)
    print("OVERALL COORDINATE RANGES (Across All Houses)")
    print("=" * 80)
    print()
    
    overall_x_min = min(all_x_min)
    overall_x_max = max(all_x_max)
    overall_y_min = min(all_y_min)
    overall_y_max = max(all_y_max)
    overall_z_min = min(all_z_min)
    overall_z_max = max(all_z_max)
    
    print(f"X Coordinate:")
    print(f"  Overall range: [{overall_x_min:7.1f}, {overall_x_max:7.1f}]")
    print(f"  Overall span:  {overall_x_max - overall_x_min:7.1f}")
    print(f"  Per-house span: min={min(all_x_ranges):6.1f}, max={max(all_x_ranges):6.1f}, avg={np.mean(all_x_ranges):6.1f}")
    print()
    
    print(f"Y Coordinate:")
    print(f"  Overall range: [{overall_y_min:7.1f}, {overall_y_max:7.1f}]")
    print(f"  Overall span:  {overall_y_max - overall_y_min:7.1f}")
    print(f"  Per-house span: min={min(all_y_ranges):6.1f}, max={max(all_y_ranges):6.1f}, avg={np.mean(all_y_ranges):6.1f}")
    print()
    
    print(f"Z Coordinate:")
    print(f"  Overall range: [{overall_z_min:7.1f}, {overall_z_max:7.1f}]")
    print(f"  Overall span:  {overall_z_max - overall_z_min:7.1f}")
    print(f"  Per-house span: min={min(all_z_ranges):6.1f}, max={max(all_z_ranges):6.1f}, avg={np.mean(all_z_ranges):6.1f}")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Houses processed: {len(all_x_min)}")
    print(f"Overall bounding box:")
    print(f"  X: [{overall_x_min:.1f}, {overall_x_max:.1f}] (span: {overall_x_max - overall_x_min:.1f})")
    print(f"  Y: [{overall_y_min:.1f}, {overall_y_max:.1f}] (span: {overall_y_max - overall_y_min:.1f})")
    print(f"  Z: [{overall_z_min:.1f}, {overall_z_max:.1f}] (span: {overall_z_max - overall_z_min:.1f})")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check coordinate ranges for all houses in dataset/3dcraft"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/3dcraft",
        help="Path to dataset directory (default: dataset/3dcraft)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset subset to check (default: train)"
    )
    parser.add_argument(
        "--max_houses",
        type=int,
        default=None,
        help="Maximum number of houses to check (default: all)"
    )
    
    args = parser.parse_args()
    
    check_coordinate_ranges(
        data_dir=args.data_dir,
        subset=args.subset,
        max_houses=args.max_houses
    )


if __name__ == "__main__":
    main()

