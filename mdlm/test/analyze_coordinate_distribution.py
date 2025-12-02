#!/usr/bin/env python3
"""Analyze the coordinate distribution of blocks in the Craft3D dataset."""

import sys
from pathlib import Path
import json

# Add parent directory to path to import helper module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from helper.craft3d_dataset import Craft3DDataset


def analyze_coordinate_distribution(dataset, output_file=None, plot=True):
    """Analyze coordinate distribution across the dataset.
    
    Args:
        dataset: Craft3DDataset instance
        output_file: Optional path to save statistics JSON
        plot: Whether to create visualization plots
    """
    print("=" * 70)
    print("COORDINATE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Collect all coordinates
    all_coords = []
    coord_ranges_per_house = []
    coord_centers_per_house = []
    
    print(f"Loading data from {dataset.subset} split...")
    print(f"Total houses: {dataset.get_num_houses()}")
    print()
    
    for i in range(dataset.get_num_houses()):
        house = dataset.get_house(i)
        coords = house[:, 1:].cpu().numpy().astype(float)  # Extract x, y, z columns
        
        all_coords.append(coords)
        
        # Per-house statistics
        if len(coords) > 0:
            coord_ranges_per_house.append({
                'x_range': [coords[:, 0].min(), coords[:, 0].max()],
                'y_range': [coords[:, 1].min(), coords[:, 1].max()],
                'z_range': [coords[:, 2].min(), coords[:, 2].max()],
                'x_span': coords[:, 0].max() - coords[:, 0].min(),
                'y_span': coords[:, 1].max() - coords[:, 1].min(),
                'z_span': coords[:, 2].max() - coords[:, 2].min(),
            })
            coord_centers_per_house.append({
                'x_center': coords[:, 0].mean(),
                'y_center': coords[:, 1].mean(),
                'z_center': coords[:, 2].mean(),
            })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{dataset.get_num_houses()} houses...")
    
    print(f"  Processed all {dataset.get_num_houses()} houses!")
    print()
    
    # Combine all coordinates
    all_coords = np.concatenate(all_coords, axis=0)  # Shape: (N, 3)
    
    print("=" * 70)
    print("COORDINATE STATISTICS")
    print("=" * 70)
    print()
    print(f"Total blocks: {len(all_coords):,}")
    print()
    
    # Overall coordinate ranges
    print("Overall coordinate ranges:")
    print(f"  X: [{all_coords[:, 0].min():.1f}, {all_coords[:, 0].max():.1f}]")
    print(f"  Y: [{all_coords[:, 1].min():.1f}, {all_coords[:, 1].max():.1f}]")
    print(f"  Z: [{all_coords[:, 2].min():.1f}, {all_coords[:, 2].max():.1f}]")
    print()
    
    print("Overall coordinate means:")
    print(f"  X: {all_coords[:, 0].mean():.2f}")
    print(f"  Y: {all_coords[:, 1].mean():.2f}")
    print(f"  Z: {all_coords[:, 2].mean():.2f}")
    print()
    
    print("Overall coordinate std dev:")
    print(f"  X: {all_coords[:, 0].std():.2f}")
    print(f"  Y: {all_coords[:, 1].std():.2f}")
    print(f"  Z: {all_coords[:, 2].std():.2f}")
    print()
    
    # Per-house statistics
    if coord_ranges_per_house:
        x_spans = [r['x_span'] for r in coord_ranges_per_house]
        y_spans = [r['y_span'] for r in coord_ranges_per_house]
        z_spans = [r['z_span'] for r in coord_ranges_per_house]
        
        print("Per-house coordinate spans:")
        print(f"  X span - Mean: {np.mean(x_spans):.2f}, Min: {min(x_spans):.1f}, Max: {max(x_spans):.1f}")
        print(f"  Y span - Mean: {np.mean(y_spans):.2f}, Min: {min(y_spans):.1f}, Max: {max(y_spans):.1f}")
        print(f"  Z span - Mean: {np.mean(z_spans):.2f}, Min: {min(z_spans):.1f}, Max: {max(z_spans):.1f}")
        print()
        
        x_centers = [c['x_center'] for c in coord_centers_per_house]
        y_centers = [c['y_center'] for c in coord_centers_per_house]
        z_centers = [c['z_center'] for c in coord_centers_per_house]
        
        print("Per-house coordinate centers:")
        print(f"  X center - Mean: {np.mean(x_centers):.2f}, Std: {np.std(x_centers):.2f}")
        print(f"  Y center - Mean: {np.mean(y_centers):.2f}, Std: {np.std(y_centers):.2f}")
        print(f"  Z center - Mean: {np.mean(z_centers):.2f}, Std: {np.std(z_centers):.2f}")
        print()
    
    # Check if coordinates are normalized (should be in 0-31 range for 32x32x32)
    expected_max = 31  # For 32x32x32 normalized schematics
    x_in_range = (all_coords[:, 0] >= 0) & (all_coords[:, 0] <= expected_max)
    y_in_range = (all_coords[:, 1] >= 0) & (all_coords[:, 1] <= expected_max)
    z_in_range = (all_coords[:, 2] >= 0) & (all_coords[:, 2] <= expected_max)
    
    print("Normalization check (expected range: 0-31 for 32x32x32):")
    print(f"  X in range: {x_in_range.sum()}/{len(all_coords)} ({100*x_in_range.sum()/len(all_coords):.1f}%)")
    print(f"  Y in range: {y_in_range.sum()}/{len(all_coords)} ({100*y_in_range.sum()/len(all_coords):.1f}%)")
    print(f"  Z in range: {z_in_range.sum()}/{len(all_coords)} ({100*z_in_range.sum()/len(all_coords):.1f}%)")
    print()
    
    # Save statistics
    stats = {
        "total_blocks": int(len(all_coords)),
        "overall_ranges": {
            "x": [float(all_coords[:, 0].min()), float(all_coords[:, 0].max())],
            "y": [float(all_coords[:, 1].min()), float(all_coords[:, 1].max())],
            "z": [float(all_coords[:, 2].min()), float(all_coords[:, 2].max())],
        },
        "overall_means": {
            "x": float(all_coords[:, 0].mean()),
            "y": float(all_coords[:, 1].mean()),
            "z": float(all_coords[:, 2].mean()),
        },
        "overall_stds": {
            "x": float(all_coords[:, 0].std()),
            "y": float(all_coords[:, 1].std()),
            "z": float(all_coords[:, 2].std()),
        },
        "normalization_check": {
            "x_in_range": int(x_in_range.sum()),
            "y_in_range": int(y_in_range.sum()),
            "z_in_range": int(z_in_range.sum()),
            "total": int(len(all_coords)),
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {output_file}")
        print()
    
    # Create visualizations
    if plot:
        print("=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)
        print()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Coordinate distributions
        axes[0, 0].hist(all_coords[:, 0], bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('X Coordinate Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(all_coords[:, 0].mean(), color='blue', linestyle='--', label=f'Mean: {all_coords[:, 0].mean():.1f}')
        axes[0, 0].legend()
        
        axes[0, 1].hist(all_coords[:, 1], bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Y Coordinate')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Y Coordinate Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(all_coords[:, 1].mean(), color='blue', linestyle='--', label=f'Mean: {all_coords[:, 1].mean():.1f}')
        axes[0, 1].legend()
        
        axes[0, 2].hist(all_coords[:, 2], bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Z Coordinate')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Z Coordinate Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axvline(all_coords[:, 2].mean(), color='red', linestyle='--', label=f'Mean: {all_coords[:, 2].mean():.1f}')
        axes[0, 2].legend()
        
        # Plot 2: 2D projections
        axes[1, 0].scatter(all_coords[:, 0], all_coords[:, 1], alpha=0.1, s=1, c='red')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        axes[1, 0].set_title('X-Y Projection')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect('equal', adjustable='box')
        
        axes[1, 1].scatter(all_coords[:, 0], all_coords[:, 2], alpha=0.1, s=1, c='green')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Z Coordinate')
        axes[1, 1].set_title('X-Z Projection')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal', adjustable='box')
        
        axes[1, 2].scatter(all_coords[:, 1], all_coords[:, 2], alpha=0.1, s=1, c='blue')
        axes[1, 2].set_xlabel('Y Coordinate')
        axes[1, 2].set_ylabel('Z Coordinate')
        axes[1, 2].set_title('Y-Z Projection')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_file.replace('.json', '_coordinates.png') if output_file else 'coordinate_distribution.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        print()
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("(Plot display not available in non-interactive environment)")
            print()
    
    return stats


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze coordinate distribution in Craft3D dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/3dcraft",
        help="Path to the Craft3D data directory"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset subset to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for statistics JSON (default: coord_stats_{subset}.json)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creating visualization plots"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of houses to analyze (for faster testing)"
    )
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"coord_stats_{args.subset}.json"
    
    print("Loading Craft3D dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Subset: {args.subset}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()
    
    try:
        # Create dataset instance
        dataset = Craft3DDataset(
            data_dir=args.data_dir,
            subset=args.subset,
            max_samples=args.max_samples,
        )
        
        # Analyze distribution
        stats = analyze_coordinate_distribution(
            dataset,
            output_file=args.output,
            plot=not args.no_plot
        )
        
        print("=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

