#!/usr/bin/env python3
"""Convert block IDs in 3dcraft_normalized dataset and print block distribution.

This script:
1. Loads an ID mapping file (default: old_id_to_compact_id.json for compact IDs)
2. Converts all block IDs in schematic.npy files from old IDs to new/compact IDs
3. Optionally saves converted schematics (with --save flag)
4. Prints block distribution statistics

Usage:
    # Convert to compact IDs (default)
    python convert_block_ids.py --save
    
    # Convert to new IDs (original old IDs)
    python convert_block_ids.py --mapping_file configs/minecraft/old_id_to_new_id.json --save
    
    # Convert in-place (overwrites original files)
    python convert_block_ids.py --mapping_file configs/minecraft/old_id_to_compact_id.json --in_place
"""

import argparse
import ast
import json
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm


def load_id_mapping(mapping_path):
    """Load the ID mapping from JSON file.
    
    Handles both JSON files with string keys and files with unquoted integer keys.
    """
    with open(mapping_path, 'r') as f:
        content = f.read()
    
    # Try JSON first (for files with string keys like old_id_to_compact_id.json)
    try:
        mapping = json.loads(content)
        # Convert string keys to integers
        return {int(k): v for k, v in mapping.items()}
    except (json.JSONDecodeError, ValueError):
        # Fall back to ast.literal_eval (for files with unquoted integer keys)
        mapping = ast.literal_eval(content)
        return mapping


def convert_schematic(schematic_path, id_mapping, save=False, output_dir=None, in_place=False):
    """Convert block IDs in a schematic file.
    
    Args:
        schematic_path: Path to schematic.npy file
        id_mapping: Dictionary mapping old_id -> new_id
        save: If True, save the converted schematic
        output_dir: Directory to save converted schematics (if save=True and not in_place)
        in_place: If True, overwrite the original file
        
    Returns:
        Counter of block IDs in the schematic
    """
    try:
        schematic = np.load(schematic_path)
    except Exception as e:
        print(f"Warning: Failed to load {schematic_path}: {e}")
        return Counter()
    
    # Handle both 3D and 4D schematics
    if len(schematic.shape) == 4:
        # Shape: (y, z, x, 2) - extract block IDs from first channel
        block_ids = schematic[..., 0].copy()  # Shape: (y, z, x)
        metadata = schematic[..., 1].copy() if schematic.shape[3] > 1 else None
    elif len(schematic.shape) == 3:
        # Shape: (y, z, x) - already just block IDs
        block_ids = schematic.copy()
        metadata = None
    else:
        print(f"Warning: Unexpected schematic shape {schematic.shape} in {schematic_path}")
        return Counter()
    
    # Get unique block IDs to convert
    unique_ids = np.unique(block_ids)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude air (0)
    
    # Convert block IDs using vectorized operation
    # Create a conversion array for all possible IDs (0-256)
    # Default to 0 (air) for IDs not in mapping
    max_id = max(int(block_ids.max()) + 1, 257)
    conversion_array = np.zeros(max_id, dtype=block_ids.dtype)
    
    # Fill in mappings from id_mapping
    for old_id, new_id in id_mapping.items():
        if 0 <= old_id < max_id:
            conversion_array[old_id] = new_id
    
    # Apply conversion using numpy fancy indexing
    # Clip block_ids to valid range
    block_ids_clipped = np.clip(block_ids, 0, max_id - 1)
    converted_block_ids = conversion_array[block_ids_clipped]
    
    # Count block IDs (excluding air)
    non_air_mask = converted_block_ids > 0
    block_id_counts = Counter(converted_block_ids[non_air_mask].flatten().tolist())
    
    # Save converted schematic if requested
    if save or in_place:
        schematic_path_obj = Path(schematic_path)
        
        if in_place:
            # Overwrite original file
            output_path = schematic_path_obj
        else:
            # Save to output directory
            if output_dir is None:
                output_dir = "dataset/3dcraft_normalized_converted"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find the house directory name
            house_dir = schematic_path_obj.parent.name
            output_path = output_dir / house_dir / "schematic.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reconstruct schematic with converted IDs
        if len(schematic.shape) == 4:
            converted_schematic = schematic.copy()
            converted_schematic[..., 0] = converted_block_ids
        else:
            converted_schematic = converted_block_ids
        
        np.save(output_path, converted_schematic)
    
    return block_id_counts


def main():
    parser = argparse.ArgumentParser(description="Convert block IDs in 3dcraft_normalized dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/3dcraft_normalized",
        help="Path to 3dcraft_normalized dataset directory"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="configs/minecraft/old_id_to_compact_id.json",
        help="Path to ID mapping file (default: old_id_to_compact_id.json for compact IDs, or old_id_to_new_id.json for new IDs)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save converted schematics to output directory"
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Modify schematics in-place (overwrite original files). Use with caution!"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/3dcraft_normalized_converted",
        help="Output directory for converted schematics (if --save is used). Ignored if --in_place is used"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of schematics to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load ID mapping
    print(f"Loading ID mapping from {args.mapping_file}...")
    mapping_path = Path(args.mapping_file)
    if not mapping_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        mapping_path = script_dir / args.mapping_file
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {args.mapping_file}")
    
    id_mapping = load_id_mapping(mapping_path)
    print(f"Loaded {len(id_mapping)} ID mappings")
    
    # Show mapping range
    unique_target_ids = sorted(set(id_mapping.values()))
    print(f"Mapping old IDs (0-256) to compact IDs: {min(unique_target_ids)} to {max(unique_target_ids)}")
    print(f"Total unique compact IDs: {len(unique_target_ids)}")
    
    # Find all schematic.npy files
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / args.data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    houses_dir = data_dir / "houses"
    if not houses_dir.exists():
        raise FileNotFoundError(f"Houses directory not found: {houses_dir}")
    
    print(f"Scanning for schematic.npy files in {houses_dir}...")
    schematic_files = list(houses_dir.glob("*/schematic.npy"))
    
    if args.limit:
        schematic_files = schematic_files[:args.limit]
        print(f"Processing limited to {len(schematic_files)} files")
    
    print(f"Found {len(schematic_files)} schematic files")
    print(f"Converting block IDs...")
    
    # Convert all schematics and collect statistics
    all_block_counts = Counter()
    failed_files = []
    
    for schematic_path in tqdm(schematic_files, desc="Converting"):
        try:
            block_counts = convert_schematic(
                schematic_path,
                id_mapping,
                save=args.save,
                output_dir=args.output_dir if args.save and not args.in_place else None,
                in_place=args.in_place
            )
            all_block_counts += block_counts
        except Exception as e:
            failed_files.append((schematic_path, str(e)))
            print(f"\nWarning: Failed to process {schematic_path}: {e}")
    
    # Get all possible compact IDs from the mapping
    unique_compact_ids_in_mapping = set(id_mapping.values())
    unique_compact_ids_sorted = sorted(unique_compact_ids_in_mapping)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("BLOCK ID DISTRIBUTION (After Conversion)")
    print("=" * 70)
    
    total_blocks = sum(all_block_counts.values())
    unique_blocks = len(all_block_counts)
    
    print(f"\nTotal blocks: {total_blocks:,}")
    print(f"Unique block types found in dataset: {unique_blocks}")
    print(f"Block ID range in dataset: {min(all_block_counts.keys()) if all_block_counts else 0} to {max(all_block_counts.keys()) if all_block_counts else 0}")
    
    # Find unused IDs
    used_ids = set(all_block_counts.keys())
    unused_ids = sorted(unique_compact_ids_in_mapping - used_ids)
    
    print(f"\nUnused compact IDs (in mapping but not in dataset): {len(unused_ids)}")
    if unused_ids:
        print(f"Unused ID range: {min(unused_ids)} to {max(unused_ids)}")
        print(f"Unused IDs: {unused_ids}")
    else:
        print("All compact IDs in mapping are used!")
    
    # Top 20 most common blocks
    print("\n" + "=" * 70)
    print("TOP 20 MOST COMMON BLOCK TYPES")
    print("=" * 70)
    print(f"{'Block ID':<12} {'Count':<15} {'Percentage':<12}")
    print("-" * 70)
    
    for block_id, count in all_block_counts.most_common(20):
        percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
        print(f"{block_id:<12} {count:<15,} {percentage:<11.2f}%")
    
    # All blocks sorted by ID
    print("\n" + "=" * 70)
    print("ALL BLOCK TYPES (sorted by ID)")
    print("=" * 70)
    print(f"{'Block ID':<12} {'Count':<15} {'Percentage':<12} {'Status':<10}")
    print("-" * 70)
    
    for block_id in unique_compact_ids_sorted:
        if block_id in all_block_counts:
            count = all_block_counts[block_id]
            percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
            print(f"{block_id:<12} {count:<15,} {percentage:<11.2f}% {'USED':<10}")
        else:
            print(f"{block_id:<12} {'0':<15} {'0.00':<11}% {'UNUSED':<10}")
    
    # Summary of unused IDs
    if unused_ids:
        print("\n" + "=" * 70)
        print("UNUSED COMPACT IDs SUMMARY")
        print("=" * 70)
        print(f"Total unused IDs: {len(unused_ids)}")
        print(f"Unused ID list: {unused_ids}")
        print(f"\nNote: These compact IDs exist in the mapping but were not found in any schematic files.")
    
    # Save statistics to file
    stats_file = Path(args.output_dir if args.save else data_dir) / "block_stats_converted.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get unused IDs for stats (reuse already calculated)
    stats = {
        "total_blocks": total_blocks,
        "unique_block_types": unique_blocks,
        "block_id_range": [min(all_block_counts.keys()) if all_block_counts else 0, max(all_block_counts.keys()) if all_block_counts else 0],
        "distribution": dict(all_block_counts),
        "unused_ids": unused_ids,
        "unused_count": len(unused_ids),
        "total_possible_ids": len(unique_compact_ids_in_mapping),
        "used_ids": sorted(list(used_ids))
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {stats_file}")
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files failed to process")
        print("First 10 failures:")
        for path, error in failed_files[:10]:
            print(f"  {path}: {error}")
    
    if args.in_place:
        print(f"\n✓ Converted schematics modified in-place in: {houses_dir}")
    elif args.save:
        print(f"\n✓ Converted schematics saved to: {args.output_dir}")
    else:
        print("\nNote: Use --save flag to save converted schematics, or --in_place to modify files in-place")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

