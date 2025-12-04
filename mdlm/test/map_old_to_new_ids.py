#!/usr/bin/env python3
"""Map old block IDs (0-255) to new IDs based on tok_to_block_new.json.

Algorithm:
1. For each old ID (0-255):
   - If the old ID is NOT in tok_to_block_new.json, map it to 0
   - If the old ID IS in tok_to_block_new.json:
     - Get the block name from tok_to_block_new.json[old_id]
     - Find the first old ID in tok_to_block_old.json that maps to the same block name
     - Map old_id -> that found old_id (the "original" ID for that block)
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


def create_old_to_new_id_mapping(tok_to_block_new_path, tok_to_block_old_path, output_path):
    """Create a mapping from old IDs (0-255) to new IDs.
    
    Args:
        tok_to_block_new_path: Path to tok_to_block_new.json
        tok_to_block_old_path: Path to tok_to_block_old.json
        output_path: Path to save the output mapping
    """
    # Load mappings
    print(f"Loading {tok_to_block_new_path}...")
    tok_to_block_new = load_json_mapping(tok_to_block_new_path)
    print(f"Loaded {len(tok_to_block_new)} entries from tok_to_block_new.json")
    
    print(f"Loading {tok_to_block_old_path}...")
    tok_to_block_old = load_json_mapping(tok_to_block_old_path)
    print(f"Loaded {len(tok_to_block_old)} entries from tok_to_block_old.json")
    
    # Create reverse mapping: block_name -> first old_id that maps to it
    # This finds the "original" ID for each block name
    block_name_to_original_old_id = {}
    for old_id, block_name in tok_to_block_old.items():
        if block_name not in block_name_to_original_old_id:
            block_name_to_original_old_id[block_name] = old_id
    
    print(f"Created reverse mapping for {len(block_name_to_original_old_id)} unique block names")
    
    # Create the old_id -> new_id mapping
    old_to_new_id = {}
    mapped_to_zero = []
    mapped_to_original = []
    
    for old_id in range(256):  # 0-255
        if old_id not in tok_to_block_new:
            # Not in new mapping, map to 0
            old_to_new_id[old_id] = 0
            mapped_to_zero.append(old_id)
        else:
            # In new mapping, find the original ID for this block name
            block_name = tok_to_block_new[old_id]
            if block_name in block_name_to_original_old_id:
                original_old_id = block_name_to_original_old_id[block_name]
                old_to_new_id[old_id] = original_old_id
                if old_id != original_old_id:
                    mapped_to_original.append((old_id, original_old_id, block_name))
            else:
                # Block name not found in old mapping, map to 0
                old_to_new_id[old_id] = 0
                mapped_to_zero.append(old_id)
                print(f"Warning: Block name '{block_name}' from old_id {old_id} not found in tok_to_block_old.json")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("MAPPING STATISTICS")
    print("=" * 70)
    print(f"Total old IDs processed: 256 (0-255)")
    print(f"IDs mapped to 0 (not in new mapping or block name not found): {len(mapped_to_zero)}")
    print(f"IDs mapped to original ID: {len(mapped_to_original)}")
    print(f"IDs that map to themselves: {256 - len(mapped_to_zero) - len(mapped_to_original)}")
    
    # Show some examples of mappings
    if mapped_to_original:
        print("\n" + "=" * 70)
        print("SAMPLE MAPPINGS (old_id -> original_id, block_name)")
        print("=" * 70)
        print(f"{'Old ID':<10} {'Original ID':<15} {'Block Name':<40}")
        print("-" * 70)
        for old_id, original_id, block_name in mapped_to_original[:20]:
            print(f"{old_id:<10} {original_id:<15} {block_name:<40}")
        if len(mapped_to_original) > 20:
            print(f"... and {len(mapped_to_original) - 20} more")
    
    # Save the mapping
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to string keys for JSON compatibility
    old_to_new_id_str = {str(k): v for k, v in old_to_new_id.items()}
    
    with open(output_path, 'w') as f:
        json.dump(old_to_new_id_str, f, indent=4)
    
    print(f"\n✓ Mapping saved to: {output_path}")
    print(f"  Format: old_id (string) -> new_id (integer)")
    print(f"  Total entries: {len(old_to_new_id)}")
    
    return old_to_new_id


def main():
    parser = argparse.ArgumentParser(
        description="Map old block IDs (0-255) to new IDs based on tok_to_block_new.json"
    )
    parser.add_argument(
        "--tok_to_block_new",
        type=str,
        default="configs/minecraft/tok_to_block_new.json",
        help="Path to tok_to_block_new.json"
    )
    parser.add_argument(
        "--tok_to_block_old",
        type=str,
        default="configs/minecraft/tok_to_block_old.json",
        help="Path to tok_to_block_old.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/minecraft/old_id_to_new_id.json",
        help="Output path for the mapping file"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if needed
    script_dir = Path(__file__).parent.parent
    
    tok_to_block_new_path = Path(args.tok_to_block_new)
    if not tok_to_block_new_path.exists():
        tok_to_block_new_path = script_dir / args.tok_to_block_new
        if not tok_to_block_new_path.exists():
            raise FileNotFoundError(f"File not found: {args.tok_to_block_new}")
    
    tok_to_block_old_path = Path(args.tok_to_block_old)
    if not tok_to_block_old_path.exists():
        tok_to_block_old_path = script_dir / args.tok_to_block_old
        if not tok_to_block_old_path.exists():
            raise FileNotFoundError(f"File not found: {args.tok_to_block_old}")
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / args.output
    
    create_old_to_new_id_mapping(
        tok_to_block_new_path,
        tok_to_block_old_path,
        output_path
    )


if __name__ == "__main__":
    main()

