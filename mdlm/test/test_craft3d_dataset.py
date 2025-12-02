#!/usr/bin/env python3
"""Test script to load and display Craft3D dataset entries."""

import sys
from pathlib import Path

# Add parent directory to path to import helper module
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from helper.craft3d_dataset import Craft3DDataset


def print_dataset_entry(dataset, house_index=0):
    """Print coordinates and block IDs for a dataset entry.
    
    Args:
        dataset: Craft3DDataset instance
        house_index: Index of the house to display (default: 0)
    """
    print(f"\n{'='*60}")
    print(f"Dataset: Craft3D")
    print(f"Subset: {dataset.subset}")
    print(f"Total houses: {dataset.get_num_houses()}")
    print(f"{'='*60}\n")
    
    if house_index >= dataset.get_num_houses():
        print(f"Error: House index {house_index} is out of range (max: {dataset.get_num_houses() - 1})")
        return
    
    # Get the house annotation
    annotation = dataset.get_house(house_index)
    
    print(f"House #{house_index}")
    print(f"Total blocks: {len(annotation)}")
    print(f"Annotation shape: {annotation.shape}")
    print(f"Annotation dtype: {annotation.dtype}")
    print(f"\n{'='*60}")
    print("Block ID | X Coordinate | Y Coordinate | Z Coordinate")
    print(f"{'-'*60}")
    
    # Print first 20 blocks (or all if less than 20)
    num_to_print = min(20, len(annotation))
    for i in range(num_to_print):
        block_id = annotation[i, 0].item()
        x = annotation[i, 1].item()
        y = annotation[i, 2].item()
        z = annotation[i, 3].item()
        print(f"  {block_id:6d} | {x:12d} | {y:12d} | {z:12d}")
    
    if len(annotation) > 20:
        print(f"\n... ({len(annotation) - 20} more blocks)")
    
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"  Block IDs range: [{annotation[:, 0].min().item()}, {annotation[:, 0].max().item()}]")
    print(f"  X coordinates range: [{annotation[:, 1].min().item()}, {annotation[:, 1].max().item()}]")
    print(f"  Y coordinates range: [{annotation[:, 2].min().item()}, {annotation[:, 2].max().item()}]")
    print(f"  Z coordinates range: [{annotation[:, 3].min().item()}, {annotation[:, 3].max().item()}]")
    
    # Show unique block types
    unique_block_ids = torch.unique(annotation[:, 0])
    print(f"  Unique block types: {len(unique_block_ids)}")
    print(f"  Block type IDs: {unique_block_ids.tolist()[:10]}{'...' if len(unique_block_ids) > 10 else ''}")
    
    # Test normalization if available
    if hasattr(dataset, '_normalize_coordinates'):
        print(f"\n{'='*60}")
        print("Testing normalization:")
        normalized = dataset._normalize_coordinates(annotation, block_size=32)
        if normalized is not None:
            print(f"  Normalized shape: {normalized.shape}")
            print(f"  Normalized X range: [{normalized[:, 1].min().item()}, {normalized[:, 1].max().item()}]")
            print(f"  Normalized Y range: [{normalized[:, 2].min().item()}, {normalized[:, 2].max().item()}]")
            print(f"  Normalized Z range: [{normalized[:, 3].min().item()}, {normalized[:, 3].max().item()}]")
        else:
            print("  Normalization returned None (empty annotation)")


def main():
    """Main function to test the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Craft3D dataset loading")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../voxelcnn/data",  # Default to voxelcnn data directory
        help="Path to the Craft3D data directory"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset subset to load"
    )
    parser.add_argument(
        "--house_index",
        type=int,
        default=0,
        help="Index of the house to display"
    )
    args = parser.parse_args()
    
    print("Loading Craft3D dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Subset: {args.subset}")
    
    try:
        # Create dataset instance
        dataset = Craft3DDataset(
            data_dir=args.data_dir,
            subset=args.subset,
            max_samples=None,  # Load all samples
        )
        
        # Print dataset entry
        print_dataset_entry(dataset, house_index=args.house_index)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

