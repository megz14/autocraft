import sys
import os
import argparse
import numpy as np
import minescript

# Mapping of numeric block IDs (0-256) to Minecraft namespaced IDs
# Based on https://minecraft-ids.grahamedgecombe.com/
BLOCK_ID_TO_NAME = {
    0: "minecraft:air",
    1: "minecraft:stone",
    2: "minecraft:dirt",
    3: "minecraft:cobblestone",
    4: "minecraft:oak_planks",
    5: "minecraft:gravel",
    6: "minecraft:oak_log",
    7: "minecraft:sponge",
    8: "minecraft:glass",
    9: "minecraft:lapis_block",
    10: "minecraft:sandstone",
    11: "minecraft:noteblock",
    12: "minecraft:white_wool",
    13: "minecraft:gold_block",
    14: "minecraft:iron_block",
    15: "minecraft:double_stone_slab",
    16: "minecraft:brick_block",
    17: "minecraft:obsidian",
    18: "minecraft:diamond_block",
    19: "minecraft:clay",
    20: "minecraft:netherrack",
    21: "minecraft:soul_sand",
    22: "minecraft:glowstone",
    23: "minecraft:white_stained_glass",
    24: "minecraft:stonebrick",
    25: "minecraft:brown_mushroom_block",
    26: "minecraft:red_mushroom_block",
    27: "minecraft:nether_brick",
    28: "minecraft:end_stone",
    29: "minecraft:lit_redstone_lamp",
    30: "minecraft:oak_planks",
    31: "minecraft:emerald_block",
    32: "minecraft:redstone_block",
    33: "minecraft:quartz_block",
    34: "minecraft:stained_hardened_clay",
    35: "minecraft:slime_block",
    36: "minecraft:prismarine",
    37: "minecraft:sea_lantern",
    38: "minecraft:hay_block",
    39: "minecraft:hardened_clay",
    40: "minecraft:coal_block",
    41: "minecraft:purpur_block",
    42: "minecraft:end_bricks",
    43: "minecraft:white_glazed_terracotta"
}


def get_block_name(block_id):
    """Convert numeric block ID to Minecraft namespaced ID."""
    if block_id < 0 or block_id > 256:
        return None
    return BLOCK_ID_TO_NAME.get(block_id, "minecraft:air")


def main():
    """
    Loads schematic.npy file and builds it in Minecraft.
    Schematic format: (y, z, x, id/meta) where id/meta is [id, meta]
    
    Usage:
        python build_schematic.py /absolute/path/to/schematic.npy
        or
        python build_schematic.py --path /absolute/path/to/schematic.npy
    """
    parser = argparse.ArgumentParser(description='Build a Minecraft structure from a schematic.npy file')
    parser.add_argument('path', nargs='?', help='Absolute or relative path to the schematic.npy file')
    parser.add_argument('--path', dest='path_arg', help='Path to the schematic.npy file (alternative to positional argument)')
    args = parser.parse_args()
    
    # Get path from either positional argument or --path flag
    schematic_path = args.path or args.path_arg
    
    # If no path provided, use default
    if not schematic_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        schematic_path = os.path.join(script_dir, "sample_output", "house-2.npy")
        print(f"No path provided, using default: {schematic_path}")
    
    # Resolve to absolute path
    if not os.path.isabs(schematic_path):
        # If relative, resolve relative to current working directory
        schematic_path = os.path.abspath(schematic_path)
    
    if not os.path.exists(schematic_path):
        print(f"Error: {schematic_path} not found!")
        return
    
    print(f"Loading schematic from: {schematic_path}")

    try:
        schematic = np.load(schematic_path, allow_pickle=True)
        
        print(f"Schematic shape: {schematic.shape}")
        print(f"Expected format: (y, z, x, id/meta) where id/meta is [id, meta]")
        
        if len(schematic.shape) != 4 or schematic.shape[3] != 2:
            print(f"Error: Expected 4D array with shape (y, z, x, 2), got {schematic.shape}")
            return

        dim_y, dim_z, dim_x, _ = schematic.shape
        print(f"Dimensions: y={dim_y}, z={dim_z}, x={dim_x}")
        print(f"Building {dim_y * dim_z * dim_x} potential blocks...")

    except Exception as e:
        print(f"Load Error: {e}")
        return

    # Build in the world
    try:
        base_offset_x = 5
        base_offset_y = 0
        base_offset_z = 0
        
        count = 0
        air_count = 0
        
        for y in range(dim_y):
            for z in range(dim_z):
                for x in range(dim_x):
                    # Get id/meta array: [id, meta]
                    id_meta = schematic[y, z, x]
                    block_id = int(id_meta[0])
                    meta = int(id_meta[1])
                    
                    # Skip air blocks
                    if block_id == 0:
                        air_count += 1
                        continue
                    
                    # Convert block ID to Minecraft namespaced ID
                    block_name = get_block_name(block_id)
                    
                    if not block_name or block_name == "minecraft:air":
                        air_count += 1
                        continue
                    
                    # Calculate relative coordinates
                    # Schematic uses (y, z, x), Minecraft uses (x, y, z)
                    rel_x = x + base_offset_x
                    rel_y = y + base_offset_y
                    rel_z = z + base_offset_z
                    
                    # Build the block
                    command = f"execute at @p run setblock ~{rel_y} ~{rel_z} ~{rel_x} {block_name}"
                    minescript.execute(command)
                    count += 1
                    
                    # Progress report every 100 blocks
                    if count % 100 == 0:
                        print(f"Placed {count} blocks...")

        # Calculate house center and teleport player to a good viewing position
        # House spans: x from base_offset_x to base_offset_x + dim_x - 1
        #              y from base_offset_y to base_offset_y + dim_y - 1
        #              z from base_offset_z to base_offset_z + dim_z - 1
        house_center_x = base_offset_x + dim_x / 2
        house_center_y = base_offset_y + dim_y / 2
        house_center_z = base_offset_z + dim_z / 2
        
        # Teleport player to a position in front of the house (offset in z direction)
        # Position: slightly in front, at ground level + a bit higher for better view
        # Using relative coordinates (~) so it's relative to player's starting position
        teleport_offset_x = int(house_center_x)
        teleport_offset_y = max(int(house_center_y) + 5, 70)  # At least 5 blocks above center, or 70 if house is low
        teleport_offset_z = int(base_offset_z) - 10  # 10 blocks in front of the house
        
        print(f"\nTeleporting player near the house...")
        teleport_command = f"tp @p ~{teleport_offset_x} ~{teleport_offset_y} ~{teleport_offset_z}"
        minescript.execute(teleport_command)
        
        # Calculate corner coordinates
        # Bottom corner (minimum coordinates)
        corner_bottom_x = base_offset_x
        corner_bottom_y = base_offset_y
        corner_bottom_z = base_offset_z
        
        # Top corner (maximum coordinates)
        corner_top_x = base_offset_x + dim_x - 1
        corner_top_y = base_offset_y + dim_y - 1
        corner_top_z = base_offset_z + dim_z - 1
        
        # Output corner coordinates to Minecraft chat
        print(f"\nBuild complete!")
        print(f"Placed {count} blocks")
        print(f"Skipped {air_count} air blocks")
        print(f"Teleported to viewing position near the house.")
        print("Look around you!")
        
        # Send corner coordinates to Minecraft chat
        minescript.execute(f'say [Build] Bottom corner (relative): ~{corner_bottom_x} ~{corner_bottom_y} ~{corner_bottom_z}')
        minescript.execute(f'say [Build] Top corner (relative): ~{corner_top_x} ~{corner_top_y} ~{corner_top_z}')
        minescript.execute(f'say [Build] Dimensions: {dim_x} x {dim_y} x {dim_z}')

    except Exception as e:
        print(f"Build Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()