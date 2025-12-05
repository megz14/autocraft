#!/usr/bin/env python3

# Copied from voxelcnn/voxelcnn/datasets.py (Craft3DDataset)

import json
import logging
import os
import tarfile
import warnings
from os import path as osp
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import torch
from torch.utils.data import Dataset


class Craft3DDataset(Dataset):
    NUM_BLOCK_TYPES = 256
    URL = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"

    @staticmethod
    def _resolve_data_dir_init(data_dir: str) -> str:
        """Resolve data directory path, handling relative paths correctly.
        
        This method tries to resolve relative paths by looking for the dataset
        directory in common locations, especially when running from Hydra output directories.
        """
        # If it's already an absolute path, use it as-is
        if osp.isabs(data_dir):
            return data_dir
        
        # Try resolving relative to current working directory first
        abs_path = osp.abspath(data_dir)
        if osp.exists(abs_path):
            return abs_path
        
        # If not found, search up the directory tree to find project root
        # and try resolving relative to that (Hydra changes cwd to output dir)
        current = os.getcwd()
        max_levels = 10  # Limit search depth
        
        for level in range(max_levels):
            # Check if current directory contains the dataset
            candidate = osp.join(current, data_dir)
            if osp.exists(candidate):
                return osp.abspath(candidate)
            
            # Look for project root indicators - check if we're in the mdlm directory
            # (which contains main.py) or if mdlm is a subdirectory
            if osp.exists(osp.join(current, "main.py")):
                # Found main.py in current directory - this is the mdlm directory
                candidate = osp.join(current, data_dir)
                if osp.exists(candidate):
                    return osp.abspath(candidate)
            elif osp.exists(osp.join(current, "mdlm", "main.py")):
                # Found mdlm subdirectory with main.py
                candidate = osp.join(current, "mdlm", data_dir)
                if osp.exists(candidate):
                    return osp.abspath(candidate)
                # Also try without mdlm prefix
                candidate = osp.join(current, data_dir)
                if osp.exists(candidate):
                    return osp.abspath(candidate)
            
            # Move up one level
            parent = osp.dirname(current)
            if parent == current:  # Reached filesystem root
                break
            current = parent
        
        # Fall back to absolute path from current directory (will be created if needed)
        return abs_path

    def __init__(
        self,
        data_dir: str,
        subset: str,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
        next_steps: int = -1,
        max_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        # Resolve data directory path properly (handle relative paths)
        self.data_dir = Craft3DDataset._resolve_data_dir_init(data_dir)
        self.subset = subset
        self.local_size = local_size
        self.global_size = global_size
        self.history = history
        self.max_local_distance = self.local_size // 2
        self.max_global_distance = self.global_size // 2
        self.next_steps = next_steps
        self.max_samples = max_samples
        self.logger = logger

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        if not self._has_raw_data():
            self._download()

        self._load_dataset()
        self._find_valid_items()

        self.print_stats()

    def print_stats(self):
        num_blocks_per_house = [len(x) for x in self._valid_indices.values()]
        ret = "\n"
        ret += f"3D Craft Dataset\n"
        ret += f"================\n"
        ret += f"  data_dir: {self.data_dir}\n"
        ret += f"  subset: {self.subset}\n"
        ret += f"  local_size: {self.local_size}\n"
        ret += f"  global_size: {self.global_size}\n"
        ret += f"  history: {self.history}\n"
        ret += f"  next_steps: {self.next_steps}\n"
        ret += f"  max_samples: {self.max_samples}\n"
        ret += f"  --------------\n"
        ret += f"  num_houses: {len(self._valid_indices)}\n"
        ret += f"  avg_blocks_per_house: {np.mean(num_blocks_per_house):.3f}\n"
        ret += f"  min_blocks_per_house: {min(num_blocks_per_house)}\n"
        ret += f"  max_blocks_per_house: {max(num_blocks_per_house)}\n"
        ret += f"  total_valid_blocks: {len(self._flattened_valid_indices)}\n"
        ret += "\n"
        self._log(ret)

    def __len__(self) -> int:
        ret = len(self._flattened_valid_indices)
        if self.max_samples is not None:
            ret = min(ret, self.max_samples)
        return ret

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        house_id, block_id = self._flattened_valid_indices[index]
        annotation = self._all_houses[house_id]
        inputs = Craft3DDataset.prepare_inputs(
            annotation[: block_id + 1],
            local_size=self.local_size,
            global_size=self.global_size,
            history=self.history,
        )
        targets = Craft3DDataset.prepare_targets(
            annotation[block_id:],
            next_steps=self.next_steps,
            local_size=self.local_size,
        )
        return inputs, targets

    def get_house(self, index: int) -> torch.Tensor:
        return self._all_houses[index]

    def get_num_houses(self) -> int:
        return len(self._all_houses)

    @staticmethod
    @torch.no_grad()
    def prepare_inputs(
        annotation: torch.Tensor,
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
    ) -> Dict[str, torch.Tensor]:
        global_inputs = Craft3DDataset._convert_to_voxels(
            annotation, size=global_size, occupancy_only=True
        )
        local_inputs = Craft3DDataset._convert_to_voxels(
            annotation, size=local_size, occupancy_only=False
        )
        if len(annotation) == 0:
            return {
                "local": local_inputs.repeat(history, 1, 1, 1),
                "global": global_inputs,
                "center": torch.zeros((3,), dtype=torch.int64),
            }

        last_coord = annotation[-1, 1:]
        center_coord = last_coord.new_full((3,), local_size // 2)
        local_history = [local_inputs]
        for i in range(len(annotation) - 1, len(annotation) - history, -1):
            if i < 0:
                local_history.append(torch.zeros_like(local_inputs))
            else:
                prev_inputs = local_history[-1].clone()
                prev_coord = annotation[i, 1:] - last_coord + center_coord
                if all((prev_coord >= 0) & (prev_coord < local_size)):
                    x, y, z = prev_coord
                    prev_inputs[:, x, y, z] = 0
                local_history.append(prev_inputs)
        local_inputs = torch.cat(local_history, dim=0)
        return {"local": local_inputs, "global": global_inputs, "center": last_coord}

    @staticmethod
    @torch.no_grad()
    def prepare_targets(
        annotation: torch.Tensor, next_steps: int = 1, local_size: int = 7
    ) -> Dict[str, torch.Tensor]:
        coords_targets = torch.full((next_steps,), -100, dtype=torch.int64)
        types_targets = coords_targets.clone()

        if len(annotation) <= 1:
            return {"coords": coords_targets, "types": types_targets}

        offsets = torch.tensor([local_size * local_size, local_size, 1])
        last_coord = annotation[0, 1:]
        center_coord = last_coord.new_full((3,), local_size // 2)

        N = min(1 + next_steps, len(annotation))
        next_types = annotation[1:N, 0].clone()
        next_coords = annotation[1:N, 1:] - last_coord + center_coord
        mask = (next_coords < 0) | (next_coords >= local_size)
        mask = mask.any(dim=1)
        next_coords = (next_coords * offsets).sum(dim=1)
        next_coords[mask] = -100
        next_types[mask] = -100

        coords_targets[: len(next_coords)] = next_coords
        types_targets[: len(next_types)] = next_types

        return {"coords": coords_targets, "types": types_targets}

    @staticmethod
    def _convert_to_voxels(
        annotation: torch.Tensor, size: int, occupancy_only: bool = False
    ) -> torch.Tensor:
        voxels_shape = (
            (1, size, size, size)
            if occupancy_only
            else (Craft3DDataset.NUM_BLOCK_TYPES, size, size, size)
        )
        if len(annotation) == 0:
            return torch.zeros(voxels_shape, dtype=torch.float32)

        annotation = annotation.clone()
        if occupancy_only:
            annotation[:, 0] = 0
        last_coord = annotation[-1, 1:]
        center_coord = last_coord.new_tensor([size // 2, size // 2, size // 2])
        annotation[:, 1:] += center_coord - last_coord
        valid_mask = (annotation[:, 1:] >= 0) & (annotation[:, 1:] < size)
        valid_mask = valid_mask.all(dim=1)
        annotation = annotation[valid_mask]
        return torch.sparse.FloatTensor(
            annotation.t(), torch.ones(len(annotation)), voxels_shape
        ).to_dense()

    def _log(self, msg: str):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def _has_raw_data(self) -> bool:
        """Check if dataset already exists (houses directory with data files and splits.json)."""
        # self.data_dir is already resolved during initialization
        data_dir_abs = osp.abspath(self.data_dir)
        houses_dir = osp.join(data_dir_abs, "houses")
        splits_path = osp.join(data_dir_abs, "splits.json")
        
        self._log(f"Dataset check: Looking for dataset in {data_dir_abs}")
        
        # Check if splits.json exists (required for loading the dataset)
        if not osp.isfile(splits_path):
            self._log(f"Dataset check: splits.json not found at {splits_path}")
            return False
        
        self._log(f"Dataset check: Found splits.json at {splits_path}")
        
        # If houses directory exists and has content, data is already there
        if not osp.isdir(houses_dir):
            self._log(f"Dataset check: houses directory not found at {houses_dir}")
            return False
        
        self._log(f"Dataset check: Found houses directory at {houses_dir}")
            
        try:
            items = os.listdir(houses_dir)
            if len(items) == 0:
                self._log(f"Dataset check: houses directory is empty at {houses_dir}")
                return False
            
            self._log(f"Dataset check: Found {len(items)} items in houses directory")
                
            # Check if at least one house has data files
            for item in items:
                house_path = osp.join(houses_dir, item)
                if osp.isdir(house_path):
                    # Check for either schematic.npy (normalized) or placed.json (original)
                    schematic_path = osp.join(house_path, "schematic.npy")
                    placed_path = osp.join(house_path, "placed.json")
                    if osp.isfile(schematic_path):
                        self._log(f"Dataset check: Found existing dataset! Houses directory: {houses_dir}, Found schematic.npy in: {item}")
                        return True
                    elif osp.isfile(placed_path):
                        self._log(f"Dataset check: Found existing dataset! Houses directory: {houses_dir}, Found placed.json in: {item}")
                        return True
            self._log(f"Dataset check: No house directories with data files found in {houses_dir} (checked {len(items)} items)")
        except (OSError, PermissionError) as e:
            self._log(f"Dataset check: Error accessing {houses_dir}: {e}")
            return False
        
        return False

    def _download(self):
        """Download dataset only if it doesn't already exist."""
        # Double-check that data doesn't already exist
        if self._has_raw_data():
            data_dir_abs = osp.abspath(self.data_dir)
            houses_dir = osp.join(data_dir_abs, "houses")
            self._log(f"Dataset already exists at {houses_dir}, skipping download and extraction.")
            return
        
        # self.data_dir is already resolved during initialization
        data_dir_abs = osp.abspath(self.data_dir)
        os.makedirs(data_dir_abs, exist_ok=True)

        tar_path = osp.join(data_dir_abs, "houses.tar.gz")
        extracted_dir = osp.join(data_dir_abs, "houses")
        
        # Download if tar doesn't exist
        if not osp.isfile(tar_path):
            self._log(f"Downloading dataset from {Craft3DDataset.URL}")
            response = requests.get(Craft3DDataset.URL, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to retrieve dataset from url: {Craft3DDataset.URL}. "
                    f"Status: {response.status_code}"
                )
            with open(tar_path, "wb") as f:
                f.write(response.content)
            self._log(f"Download complete. Saved to {tar_path}")
        else:
            self._log(f"Archive already exists at {tar_path}, skipping download.")

        # Extract if not already extracted (check again after potential download)
        if self._has_raw_data():
            self._log(f"Dataset already extracted at {extracted_dir}, skipping extraction.")
            return
        
        self._log(f"Extracting dataset to {extracted_dir}")
        tar = tarfile.open(tar_path, "r")
        tar.extractall(data_dir_abs)
        tar.close()
        self._log(f"Extraction complete.")

    def _load_dataset(self):
        splits_path = osp.join(self.data_dir, "splits.json")
        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self._all_houses = []
        max_len = 0
        for filename in splits[self.subset]:
            schematic_path = osp.join(self.data_dir, "houses", filename, "schematic.npy")
            if not osp.isfile(schematic_path):
                warnings.warn(f"No schematic file for: {schematic_path}")
                continue
            annotation = self._load_schematic(schematic_path)
            if annotation is not None and len(annotation) >= 100:
                self._all_houses.append(annotation)
                max_len = max(max_len, len(annotation))

        if self.next_steps <= 0:
            self.next_steps = max_len

    def _load_schematic(self, schematic_path: str) -> Optional[torch.Tensor]:
        """Load schematic from normalized schematic.npy file and extract occupied blocks.
        
        Args:
            schematic_path: Path to schematic.npy file
            
        Returns:
            torch.Tensor: Shape (N, 4) where each row is [block_type, x, y, z] for occupied blocks
                         Returns None if schematic is empty or invalid
        """
        try:
            schematic = np.load(schematic_path)
        except Exception as e:
            warnings.warn(f"Failed to load schematic {schematic_path}: {e}")
            return None
        
        # Handle both 3D and 4D schematics
        if len(schematic.shape) == 4:
            # Shape: (y, z, x, 2) - extract block IDs from first channel
            block_ids = schematic[..., 0]  # Shape: (y, z, x)
        elif len(schematic.shape) == 3:
            # Shape: (y, z, x) - already just block IDs
            block_ids = schematic
        else:
            warnings.warn(f"Unexpected schematic shape: {schematic.shape}")
            return None
        
        # Find all occupied blocks (block_id > 0)
        occupied = np.where(block_ids > 0)
        
        if len(occupied[0]) == 0:
            return None
        
        # Extract block types and coordinates
        # schematic uses (y, z, x) but output format needs (x, y, z)
        types_and_coords = []
        for i in range(len(occupied[0])):
            y, z, x = occupied[0][i], occupied[1][i], occupied[2][i]
            block_type = int(block_ids[y, z, x])
            # Output format: [block_type, x, y, z]
            types_and_coords.append((block_type, x, y, z))
        
        if len(types_and_coords) == 0:
            return None
        
        return torch.tensor(types_and_coords, dtype=torch.int64)
    
    def _load_annotation(self, annotation_path: str) -> torch.Tensor:
        """Legacy method: Load from placed.json (kept for backward compatibility).
        
        Args:
            annotation_path: Path to placed.json file
            
        Returns:
            torch.Tensor: Shape (N, 4) where each row is [block_type, x, y, z]
        """
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        final_house = {}
        types_and_coords = []
        last_timestamp = -1
        for i, item in enumerate(annotation):
            timestamp, annotator_id, coordinate, block_info, action = item
            assert timestamp >= last_timestamp
            last_timestamp = timestamp
            coordinate = tuple(np.asarray(coordinate).astype(np.int64).tolist())
            block_type = np.asarray(block_info, dtype=np.uint8).astype(np.int64)[0]
            if action == "B":
                final_house.pop(coordinate, None)
            else:
                final_house[coordinate] = i
            types_and_coords.append((block_type,) + coordinate)
        indices = sorted(final_house.values())
        types_and_coords = [types_and_coords[i] for i in indices]
        return torch.tensor(types_and_coords, dtype=torch.int64)

    def _normalize_coordinates(self, annotation: torch.Tensor, block_size: int = 32) -> Optional[torch.Tensor]:
        """Normalize coordinates by centering around center of mass and resizing to block_size.
        
        Args:
            annotation (torch.Tensor): Tensor of shape (N, 4) where each row is [block_type, x, y, z]
            block_size (int): Target size for the normalized coordinates
            
        Returns:
            torch.Tensor: Normalized annotation with coordinates centered at (0,0,0) and within [0, block_size),
                         or None if empty
        """
        if len(annotation) == 0:
            return None
        
        # Extract coordinates (x, y, z)
        coords = annotation[:, 1:].float()  # Shape: (N, 3)
        block_types = annotation[:, 0]  # Shape: (N,)
        
        # Calculate center of mass
        center_of_mass = coords.mean(dim=0)  # Shape: (3,)
        
        # Center coordinates around (0, 0, 0)
        centered_coords = coords - center_of_mass
        
        # Find the bounding box
        min_coords = centered_coords.min(dim=0)[0]
        max_coords = centered_coords.max(dim=0)[0]
        
        # Calculate scale to fit within block_size
        coord_range = max_coords - min_coords
        max_range = coord_range.max().item()
        
        if max_range > 0:
            # Scale to fit within block_size (with some padding)
            scale = (block_size - 2) / max_range  # Leave 1 voxel padding on each side
            centered_coords = centered_coords * scale
        
        # Shift to positive coordinates (centered in block_size cube)
        offset = torch.tensor([block_size / 2, block_size / 2, block_size / 2], dtype=torch.float32)
        normalized_coords = centered_coords + offset
        
        # Clip to valid range [0, block_size)
        normalized_coords = torch.clamp(normalized_coords, 0, block_size - 1)
        
        # Round to integers
        normalized_coords = normalized_coords.round().long()
        
        # Combine block types and normalized coordinates
        normalized_annotation = torch.cat([
            block_types.unsqueeze(1),
            normalized_coords
        ], dim=1)
        
        return normalized_annotation

    def _resize_schematic(self, schematic: np.ndarray, block_size: int = 32) -> Optional[np.ndarray]:
        """Centers the schematic around the center of mass, resizes it to the block size and pads with zeroes.

        The normalization process:
        1. Find center of mass of occupied blocks
        2. Shift coordinates so center of mass is at (block_size//2, block_size//2, block_size//2)
        3. Crop if larger than block_size (keeps blocks closest to center)
        4. Pad with zeros if smaller than block_size
        
        This ensures all schematics are exactly block_size x block_size x block_size,
        with coordinates centered around (0,0,0) relative to the center of mass.

        Args:
            schematic (np.ndarray): The schematic - of format: (y, z, x, entryshape)
            block_size (int): Target size for each dimension (default: 32)

        Returns:
            np.ndarray: the normalized schematic of shape (block_size, block_size, block_size, entryshape)
        """
        if schematic.sum() <= 0:
            return None
        
        # Get original shape
        orig_shape = schematic.shape[:3]  # (y, z, x)
        
        # Find center of mass of occupied blocks
        # Extract first channel for occupancy check (handles both 3D and 4D schematics)
        if len(schematic.shape) == 4:
            occupancy = schematic[..., 0] > 0  # Shape: (y, z, x)
        else:
            occupancy = schematic > 0  # Shape: (y, z, x)
        
        # Get coordinates of occupied blocks
        nonzero_coords = np.nonzero(occupancy)  # Returns tuple of 3 arrays: (y_coords, z_coords, x_coords)
        if len(nonzero_coords[0]) == 0:
            return None
        
        # Stack coordinates: shape (3, N) where each column is (y, z, x)
        coords = np.stack(nonzero_coords, axis=0)  # Shape: (3, N)
        center_of_mass = np.mean(coords, axis=1)  # Average across all occupied blocks, shape: (3,)
        center_of_mass = np.round(center_of_mass).astype(int)
        
        # Calculate shift to center the schematic
        # We want center_of_mass to be at (block_size//2, block_size//2, block_size//2)
        target_center = block_size // 2
        shift = target_center - center_of_mass  # How much to shift each axis
        
        # Create new empty schematic of target size
        if len(schematic.shape) == 4:
            normalized = np.zeros((block_size, block_size, block_size, schematic.shape[3]), dtype=schematic.dtype)
        else:
            normalized = np.zeros((block_size, block_size, block_size), dtype=schematic.dtype)
        
        # Copy blocks to new positions (with shift applied)
        for y in range(orig_shape[0]):
            for z in range(orig_shape[1]):
                for x in range(orig_shape[2]):
                    # Calculate new position after shifting
                    new_y = y + shift[0]
                    new_z = z + shift[1]
                    new_x = x + shift[2]
                    
                    # Only copy if within bounds
                    if (0 <= new_y < block_size and 
                        0 <= new_z < block_size and 
                        0 <= new_x < block_size):
                        if len(schematic.shape) == 4:
                            normalized[new_y, new_z, new_x] = schematic[y, z, x]
                        else:
                            normalized[new_y, new_z, new_x] = schematic[y, z, x]
        
        return normalized

    def _find_valid_items(self):
        self._valid_indices = {}
        for i, annotation in enumerate(self._all_houses):
            diff_coord = annotation[:-1, 1:] - annotation[1:, 1:]
            valids = abs(diff_coord) <= self.max_local_distance
            valids = valids.all(dim=1).nonzero(as_tuple=True)[0]
            self._valid_indices[i] = valids.tolist()

        self._flattened_valid_indices = []
        for i, indices in self._valid_indices.items():
            for j in indices:
                self._flattened_valid_indices.append((i, j))

