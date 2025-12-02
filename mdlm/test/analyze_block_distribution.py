#!/usr/bin/env python3
"""Analyze the distribution of block IDs in the Craft3D dataset."""

import sys
from pathlib import Path
import json

# Add parent directory to path to import helper module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from collections import Counter
import matplotlib.pyplot as plt
from helper.craft3d_dataset import Craft3DDataset


def analyze_block_distribution(dataset, output_file=None, plot=True):
    """Analyze block ID distribution across the dataset.
    
    Args:
        dataset: Craft3DDataset instance
        output_file: Optional path to save statistics JSON
        plot: Whether to create visualization plots
    """
    print("=" * 70)
    print("BLOCK ID DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Collect all block IDs
    all_block_ids = []
    block_counts_per_house = []
    
    print(f"Loading data from {dataset.subset} split...")
    print(f"Total houses: {dataset.get_num_houses()}")
    print()
    
    for i in range(dataset.get_num_houses()):
        house = dataset.get_house(i)
        block_ids = house[:, 0].cpu().numpy()  # Extract block_type column
        all_block_ids.extend(block_ids.tolist())
        block_counts_per_house.append(len(block_ids))
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{dataset.get_num_houses()} houses...")
    
    print(f"  Processed all {dataset.get_num_houses()} houses!")
    print()
    
    # Calculate statistics
    all_block_ids = np.array(all_block_ids)
    block_counter = Counter(all_block_ids)
    
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print()
    print(f"Total blocks: {len(all_block_ids):,}")
    print(f"Unique block types: {len(block_counter)}")
    print(f"Block ID range: [{all_block_ids.min()}, {all_block_ids.max()}]")
    print()
    print(f"Houses analyzed: {len(block_counts_per_house)}")
    print(f"Average blocks per house: {np.mean(block_counts_per_house):.2f}")
    print(f"Min blocks per house: {min(block_counts_per_house)}")
    print(f"Max blocks per house: {max(block_counts_per_house)}")
    print()
    
    # Top block types
    print("=" * 70)
    print("TOP 20 MOST COMMON BLOCK TYPES")
    print("=" * 70)
    print()
    print(f"{'Block ID':<12} {'Count':<15} {'Percentage':<15}")
    print("-" * 42)
    
    sorted_blocks = sorted(block_counter.items(), key=lambda x: x[1], reverse=True)
    total_blocks = len(all_block_ids)
    
    for block_id, count in sorted_blocks[:20]:
        percentage = (count / total_blocks) * 100
        print(f"{block_id:<12} {count:<15,} {percentage:>6.2f}%")
    
    if len(sorted_blocks) > 20:
        print(f"\n... and {len(sorted_blocks) - 20} more block types")
    
    print()
    
    # All block types
    print("=" * 70)
    print("ALL BLOCK TYPES (sorted by ID)")
    print("=" * 70)
    print()
    print(f"{'Block ID':<12} {'Count':<15} {'Percentage':<15}")
    print("-" * 42)
    
    for block_id in sorted(block_counter.keys()):
        count = block_counter[block_id]
        percentage = (count / total_blocks) * 100
        print(f"{block_id:<12} {count:<15,} {percentage:>6.2f}%")
    
    print()
    
    # Save statistics
    stats = {
        "total_blocks": int(len(all_block_ids)),
        "unique_block_types": len(block_counter),
        "block_id_range": [int(all_block_ids.min()), int(all_block_ids.max())],
        "houses_analyzed": len(block_counts_per_house),
        "avg_blocks_per_house": float(np.mean(block_counts_per_house)),
        "min_blocks_per_house": int(min(block_counts_per_house)),
        "max_blocks_per_house": int(max(block_counts_per_house)),
        "block_distribution": {str(k): int(v) for k, v in sorted(block_counter.items())},
        "top_blocks": {str(k): int(v) for k, v in sorted_blocks[:20]}
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
        
        # Figure 1: Distribution of block IDs (bar chart)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top 20 block types
        top_20 = sorted_blocks[:20]
        block_ids_top = [b[0] for b in top_20]
        counts_top = [b[1] for b in top_20]
        
        axes[0, 0].bar(block_ids_top, counts_top, color='steelblue')
        axes[0, 0].set_xlabel('Block ID')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Top 20 Most Common Block Types')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].ticklabel_format(style='plain', axis='y')
        
        # Plot 2: All block types (sorted by ID)
        all_block_ids_sorted = sorted(block_counter.keys())
        all_counts = [block_counter[b] for b in all_block_ids_sorted]
        
        axes[0, 1].bar(all_block_ids_sorted, all_counts, color='coral', width=0.8)
        axes[0, 1].set_xlabel('Block ID')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('All Block Types Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].ticklabel_format(style='plain', axis='y')
        
        # Plot 3: Blocks per house distribution
        axes[1, 0].hist(block_counts_per_house, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Blocks per House')
        axes[1, 0].set_ylabel('Number of Houses')
        axes[1, 0].set_title('Distribution of Blocks per House')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(np.mean(block_counts_per_house), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(block_counts_per_house):.1f}')
        axes[1, 0].legend()
        
        # Plot 4: Cumulative distribution of block types
        sorted_by_count = sorted(block_counter.items(), key=lambda x: x[1], reverse=True)
        cumulative_counts = np.cumsum([b[1] for b in sorted_by_count])
        cumulative_percentage = (cumulative_counts / total_blocks) * 100
        
        axes[1, 1].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage, 
                       color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Block Type Rank (by frequency)')
        axes[1, 1].set_ylabel('Cumulative Percentage (%)')
        axes[1, 1].set_title('Cumulative Distribution of Block Types')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
        axes[1, 1].axhline(80, color='orange', linestyle='--', alpha=0.5, label='80%')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_file.replace('.json', '_distribution.png') if output_file else 'block_distribution.png'
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
    
    parser = argparse.ArgumentParser(description="Analyze block ID distribution in Craft3D dataset")
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
        help="Output file path for statistics JSON (default: block_stats_{subset}.json)"
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
        args.output = f"block_stats_{args.subset}.json"
    
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
        stats = analyze_block_distribution(
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

