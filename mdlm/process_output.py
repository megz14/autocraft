import numpy as np
import sys

blocks_path = "/workspace/autocraft/autocraft/mdlm/outputs/openwebtext/2025.11.27/235721/generated_blocks.npy"
coords_path = "/workspace/autocraft/autocraft/mdlm/outputs/openwebtext/2025.11.27/235721/coords.npy"
blocks_output = np.load(blocks_path)
coords_output = np.load(coords_path)
print("block output shape: ",blocks_output.shape)
print("block coords shape", coords_output.shape)
print("first 10")
for i in range(50):
    print(f"{i}   : blocks_output {blocks_output[:, i]} coords_output {coords_output[:, i, :]}")


blocks_path = "/workspace/autocraft/autocraft/mdlm/outputs/openwebtext/2025.11.27/200557/groundtruth_blocks.npy"
coords_path = "/workspace/autocraft/autocraft/mdlm/outputs/openwebtext/2025.11.27/200557/groundtruth_coords.npy"
blocks_output = np.load(blocks_path)
coords_output = np.load(coords_path)
print("\n\ngroundtruth")
print("block output shape: ",blocks_output.shape)
print("block coords shape", coords_output.shape)
print("first 10")
for i in range(50):
    print(f"{i}   : blocks_output {blocks_output[:, i]} coords_output {coords_output[:, i, :]}")