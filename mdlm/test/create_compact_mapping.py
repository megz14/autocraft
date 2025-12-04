#!/usr/bin/env python3
"""Create compact ID mappings from old_id_to_new_id.json.

This script:
1. Reads old_id_to_new_id.json to get unique target IDs
2. Creates a compact mapping: old_id -> compact_id (0 to num_unique-1)
3. Creates a mapping: compact_id -> block_name

Algorithm:
- Get all unique values from old_id_to_new_id.json
- Sort them and assign compact IDs (0, 1, 2, ...)
- Map each old_id to its corresponding compact_id
- For each compact_id, find the block name from tok_to_block_old.json
"""

import argparse
import ast
import json
from pathlib import Path


def load_json_mapping(file_path):
    """Load a JSON mapping file that uses integer keys (not quoted)."""
    with open(file_path, 'r') as f:
        # Use ast.literal_eval to handle unquoted integer keys
        mapping = ast.literal_eval(f.read())
    return mapping


def load_json_with_string_keys(file_path):
    """Load a JSON mapping file with string keys."""
    with open(file_path, 'r') as f:
        mapping = json.load(f)
    # Convert string keys to integers
    return {int(k): v for k, v in mapping.items()}


def create_compact_mappings(
    old_id_to_new_id_path,
    tok_to_block_old_path,
    old_id_to_compact_id_output,
    compact_id_to_block_name_output
):
    """Create compact ID mappings.
    
    Args:
        old_id_to_new_id_path: Path to old_id_to_new_id.json
        tok_to_block_old_path: Path to tok_to_block_old.json
        old_id_to_compact_id_output: Output path for old_id -> compact_id mapping
        compact_id_to_block_name_output: Output path for compact_id -> block_name mapping
    """
    # Load mappings
    print(f"Loading {old_id_to_new_id_path}...")
    old_id_to_new_id = load_json_with_string_keys(old_id_to_new_id_path)
    print(f"Loaded {len(old_id_to_new_id)} entries from old_id_to_new_id.json")
    
    print(f"Loading {tok_to_block_old_path}...")
    tok_to_block_old = load_json_mapping(tok_to_block_old_path)
    print(f"Loaded {len(tok_to_block_old)} entries from tok_to_block_old.json")
    
    # Get unique target IDs (the "new_id" values) and sort them
    unique_target_ids = sorted(set(old_id_to_new_id.values()))
    print(f"\nFound {len(unique_target_ids)} unique target IDs")
    print(f"Target ID range: {min(unique_target_ids)} to {max(unique_target_ids)}")
    
    # Create mapping: target_id -> compact_id
    target_id_to_compact_id = {target_id: compact_id for compact_id, target_id in enumerate(unique_target_ids)}
    
    print(f"Created compact ID mapping: 0 to {len(unique_target_ids) - 1}")
    
    # Create old_id -> compact_id mapping
    old_id_to_compact_id = {}
    for old_id in range(256):
        target_id = old_id_to_new_id.get(old_id, 0)
        compact_id = target_id_to_compact_id[target_id]
        old_id_to_compact_id[old_id] = compact_id
    
    # Create compact_id -> block_name mapping
    compact_id_to_block_name = {}
    missing_block_names = []
    
    for compact_id, target_id in enumerate(unique_target_ids):
        if target_id in tok_to_block_old:
            block_name = tok_to_block_old[target_id]
            compact_id_to_block_name[compact_id] = block_name
        else:
            # If target_id is 0, it's air
            if target_id == 0:
                compact_id_to_block_name[compact_id] = "minecraft:air"
            else:
                missing_block_names.append(target_id)
                compact_id_to_block_name[compact_id] = f"unknown_block_{target_id}"
    
    # Print statistics
    print("\n" + "=" * 70)
    print("COMPACT MAPPING STATISTICS")
    print("=" * 70)
    print(f"Total old IDs: 256 (0-255)")
    print(f"Unique target IDs: {len(unique_target_ids)}")
    print(f"Compact ID range: 0 to {len(unique_target_ids) - 1}")
    
    if missing_block_names:
        print(f"\nWarning: {len(missing_block_names)} target IDs not found in tok_to_block_old.json:")
        print(f"  Missing IDs: {missing_block_names}")
    else:
        print("\n✓ All target IDs have corresponding block names")
    
    # Show some sample mappings
    print("\n" + "=" * 70)
    print("SAMPLE COMPACT MAPPINGS")
    print("=" * 70)
    print(f"{'Old ID':<10} {'Target ID':<12} {'Compact ID':<12} {'Block Name':<40}")
    print("-" * 70)
    
    sample_old_ids = [0, 1, 2, 3, 4, 5, 12, 13, 17, 19, 20, 24, 25, 35, 41, 42, 45]
    for old_id in sample_old_ids[:15]:
        target_id = old_id_to_new_id.get(old_id, 0)
        compact_id = old_id_to_compact_id[old_id]
        block_name = compact_id_to_block_name[compact_id]
        print(f"{old_id:<10} {target_id:<12} {compact_id:<12} {block_name:<40}")
    
    # Save old_id -> compact_id mapping
    old_id_to_compact_id_output = Path(old_id_to_compact_id_output)
    old_id_to_compact_id_output.parent.mkdir(parents=True, exist_ok=True)
    
    old_id_to_compact_id_str = {str(k): v for k, v in old_id_to_compact_id.items()}
    with open(old_id_to_compact_id_output, 'w') as f:
        json.dump(old_id_to_compact_id_str, f, indent=4)
    
    print(f"\n✓ Saved old_id -> compact_id mapping to: {old_id_to_compact_id_output}")
    print(f"  Format: old_id (string) -> compact_id (integer)")
    print(f"  Total entries: {len(old_id_to_compact_id)}")
    
    # Save compact_id -> block_name mapping
    compact_id_to_block_name_output = Path(compact_id_to_block_name_output)
    compact_id_to_block_name_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Use integer keys for compact_id_to_block_name (like tok_to_block.json format)
    compact_id_to_block_name_int_keys = {int(k): v for k, v in compact_id_to_block_name.items()}
    
    # Write with unquoted integer keys (like tok_to_block.json)
    with open(compact_id_to_block_name_output, 'w') as f:
        f.write('{\n')
        items = sorted(compact_id_to_block_name_int_keys.items())
        for i, (compact_id, block_name) in enumerate(items):
            comma = ',' if i < len(items) - 1 else ''
            f.write(f'    {compact_id}: "{block_name}"{comma}\n')
        f.write('}\n')
    
    print(f"✓ Saved compact_id -> block_name mapping to: {compact_id_to_block_name_output}")
    print(f"  Format: compact_id (integer) -> block_name (string)")
    print(f"  Total entries: {len(compact_id_to_block_name)}")
    
    # Show compact ID distribution
    print("\n" + "=" * 70)
    print("COMPACT ID DISTRIBUTION")
    print("=" * 70)
    from collections import Counter
    compact_id_counts = Counter(old_id_to_compact_id.values())
    print(f"{'Compact ID':<12} {'Count (old IDs)':<20} {'Block Name':<40}")
    print("-" * 70)
    for compact_id in sorted(compact_id_counts.keys()):
        count = compact_id_counts[compact_id]
        block_name = compact_id_to_block_name[compact_id]
        print(f"{compact_id:<12} {count:<20} {block_name:<40}")
    
    return old_id_to_compact_id, compact_id_to_block_name


def main():
    parser = argparse.ArgumentParser(
        description="Create compact ID mappings from old_id_to_new_id.json"
    )
    parser.add_argument(
        "--old_id_to_new_id",
        type=str,
        default="configs/minecraft/old_id_to_new_id.json",
        help="Path to old_id_to_new_id.json"
    )
    parser.add_argument(
        "--tok_to_block_old",
        type=str,
        default="configs/minecraft/tok_to_block_old.json",
        help="Path to tok_to_block_old.json"
    )
    parser.add_argument(
        "--old_id_to_compact_id_output",
        type=str,
        default="configs/minecraft/old_id_to_compact_id.json",
        help="Output path for old_id -> compact_id mapping"
    )
    parser.add_argument(
        "--compact_id_to_block_name_output",
        type=str,
        default="configs/minecraft/compact_id_to_block_name.json",
        help="Output path for compact_id -> block_name mapping"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if needed
    script_dir = Path(__file__).parent.parent
    
    old_id_to_new_id_path = Path(args.old_id_to_new_id)
    if not old_id_to_new_id_path.exists():
        old_id_to_new_id_path = script_dir / args.old_id_to_new_id
        if not old_id_to_new_id_path.exists():
            raise FileNotFoundError(f"File not found: {args.old_id_to_new_id}")
    
    tok_to_block_old_path = Path(args.tok_to_block_old)
    if not tok_to_block_old_path.exists():
        tok_to_block_old_path = script_dir / args.tok_to_block_old
        if not tok_to_block_old_path.exists():
            raise FileNotFoundError(f"File not found: {args.tok_to_block_old}")
    
    old_id_to_compact_id_output = Path(args.old_id_to_compact_id_output)
    if not old_id_to_compact_id_output.is_absolute():
        old_id_to_compact_id_output = script_dir / args.old_id_to_compact_id_output
    
    compact_id_to_block_name_output = Path(args.compact_id_to_block_name_output)
    if not compact_id_to_block_name_output.is_absolute():
        compact_id_to_block_name_output = script_dir / args.compact_id_to_block_name_output
    
    create_compact_mappings(
        old_id_to_new_id_path,
        tok_to_block_old_path,
        old_id_to_compact_id_output,
        compact_id_to_block_name_output
    )


if __name__ == "__main__":
    main()

