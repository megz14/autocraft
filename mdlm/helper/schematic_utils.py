"""Utility functions for creating and manipulating schematic files."""

import numpy as np


def create_tower_schematic(block_size=32, center_x=16, center_z=16, height=20, width=3, block_id=1):
    """Create a tower schematic in the same format as schematic.npy files.
    
    Args:
        block_size: Size of the voxel grid (default: 32)
        center_x: X coordinate of tower center (default: 16)
        center_z: Z coordinate of tower center (default: 16)
        height: Height of the tower (default: 20)
        width: Width of the tower (radius from center, default: 3)
        block_id: Block ID to use for the tower (default: 1)
        
    Returns:
        numpy array of shape (block_size, block_size, block_size, 2) with tower blocks
    """
    schematic = np.zeros((block_size, block_size, block_size, 2), dtype=np.uint8)
    
    # Create a tower (vertical column)
    # Start from y=0 and go up to height
    for y in range(min(height, block_size)):
        # Create a square base
        for dx in range(-width, width + 1):
            for dz in range(-width, width + 1):
                x = center_x + dx
                z = center_z + dz
                
                # Check bounds
                if 0 <= x < block_size and 0 <= z < block_size:
                    # Place block at (y, z, x)
                    schematic[y, z, x, 0] = block_id
                    schematic[y, z, x, 1] = 0
    
    return schematic


def create_well_schematic(block_size=32, center_x=16, center_z=16, outer_radius=5, inner_radius=3, height=15, block_id=1):
    """Create a well schematic in the same format as schematic.npy files.
    
    A well has:
    - Outer circular wall
    - Inner hollow space (well hole)
    - Can extend above ground (well structure)
    
    Args:
        block_size: Size of the voxel grid (default: 32)
        center_x: X coordinate of well center (default: 16)
        center_z: Z coordinate of well center (default: 16)
        outer_radius: Outer radius of the well (default: 5)
        inner_radius: Inner radius (hollow space, default: 3)
        height: Height of the well structure (default: 15)
        block_id: Block ID to use for the well (default: 1)
        
    Returns:
        numpy array of shape (block_size, block_size, block_size, 2) with well blocks
    """
    schematic = np.zeros((block_size, block_size, block_size, 2), dtype=np.uint8)
    
    # Create a well (hollow circular structure)
    # Start from y=0 and go up to height
    for y in range(min(height, block_size)):
        # Iterate over all positions in a square around the center
        for dx in range(-outer_radius, outer_radius + 1):
            for dz in range(-outer_radius, outer_radius + 1):
                x = center_x + dx
                z = center_z + dz
                
                # Check bounds
                if 0 <= x < block_size and 0 <= z < block_size:
                    # Calculate distance from center
                    distance = np.sqrt(dx**2 + dz**2)
                    
                    # Place block if it's in the outer ring (between inner and outer radius)
                    if inner_radius <= distance <= outer_radius:
                        # Place block at (y, z, x)
                        schematic[y, z, x, 0] = block_id
                        schematic[y, z, x, 1] = 0
    
    return schematic


def load_coords_from_schematic(schematic_path, block_size=32):
    """Load coordinates from a schematic file and convert to (x, y, z) format.
    
    Args:
        schematic_path: Path to schematic.npy file
        block_size: Size of the voxel grid (default: 32)
        
    Returns:
        tuple: (coords, block_ids) where:
            - coords: numpy array of shape (N, 3) with (x, y, z) coordinates
            - block_ids: numpy array of shape (N,) with block IDs
    """
    schematic = np.load(schematic_path)
    
    # Extract block IDs (channel 0)
    if len(schematic.shape) == 4:
        block_ids_array = schematic[..., 0]  # Shape: (y, z, x)
    else:
        block_ids_array = schematic
    
    # Find all occupied blocks
    occupied = np.where(block_ids_array > 0)
    
    if len(occupied[0]) == 0:
        return np.array([]), np.array([])
    
    # Extract coordinates
    # schematic uses (y, z, x) format
    y_coords = occupied[0]
    z_coords = occupied[1]
    x_coords = occupied[2]
    block_ids = block_ids_array[occupied]
    
    # Convert to (x, y, z) format and center at (0, 0, 0)
    coords_list = []
    block_ids_list = []
    
    offset = block_size // 2
    
    for i in range(len(x_coords)):
        x = x_coords[i] - offset  # Center at 0
        y = y_coords[i] - offset
        z = z_coords[i] - offset
        block_id = block_ids[i]
        
        coords_list.append([x, y, z])
        block_ids_list.append(block_id)
    
    coords = np.array(coords_list, dtype=np.float32)
    block_ids = np.array(block_ids_list, dtype=np.int64)
    
    return coords, block_ids

