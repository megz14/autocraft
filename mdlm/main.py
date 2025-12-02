import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import numpy as np
from pathlib import Path

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


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


def _blocks_to_schematic(block_ids, coords, attention_mask=None, pad_token_id=0, block_size=32):
  """Convert block IDs and coordinates to a voxel grid schematic.
  
  Coordinates are centered at (0,0,0) and mapped to [0, block_size) range.
  Matches normalized schematic.npy format: (y, z, x, 2) where channel 0 is block IDs.
  
  Args:
    block_ids: Tensor of shape (seq_len,) containing block IDs
    coords: Tensor of shape (seq_len, 3) containing (x, y, z) coordinates (centered at 0,0,0)
    attention_mask: Optional tensor of shape (seq_len,) to filter padding
    pad_token_id: Padding token ID to filter out
    block_size: Size of the voxel grid (default: 32)
    
  Returns:
    numpy array of shape (block_size, block_size, block_size, 2) matching schematic.npy format
    Format: (y, z, x, 2) where channel 0 is block IDs, channel 1 is metadata (set to 0)
  """
  # Filter out padding positions
  if attention_mask is not None:
    valid_mask = (attention_mask == 1)
    block_ids = block_ids[valid_mask]
    coords = coords[valid_mask]
  else:
    # Filter out padding tokens
    valid_mask = (block_ids != pad_token_id)
    block_ids = block_ids[valid_mask]
    coords = coords[valid_mask]
  
  # Convert to numpy
  if isinstance(block_ids, torch.Tensor):
    block_ids = block_ids.cpu().numpy()
  if isinstance(coords, torch.Tensor):
    coords = coords.cpu().numpy()
  
  # Create empty voxel grid (y, z, x, 2) format like normalized schematic.npy
  schematic = np.zeros((block_size, block_size, block_size, 2), dtype=np.uint8)
  
  # Determine if coordinates need shifting based on their range
  # Normalized coordinates from dataset are in [0, block_size) range
  # But if they're centered at (0,0,0), they would be in [-block_size/2, block_size/2] range
  needs_shift = False
  if len(coords) > 0:
    min_coord = float(coords.min())
    max_coord = float(coords.max())
    # If we see negative coordinates, they're centered at 0 and need shifting
    if min_coord < 0:
      needs_shift = True
      offset = block_size // 2
    # If coordinates are in [0, block_size) range, use directly
    elif min_coord >= 0 and max_coord < block_size:
      needs_shift = False
      offset = 0
    else:
      # Default: assume coordinates need to be centered
      needs_shift = True
      offset = block_size // 2
  else:
    offset = 0
  
  # Place blocks in the grid
  placed_count = 0
  collision_count = 0
  out_of_bounds_count = 0
  
  for i in range(len(block_ids)):
    block_id = int(block_ids[i])
    x, y, z = coords[i]
    
    # Apply offset if needed
    if needs_shift:
      x = x + offset
      y = y + offset
      z = z + offset
    
    # Round to integers
    x = int(np.round(x))
    y = int(np.round(y))
    z = int(np.round(z))
    
    # Check bounds before clipping
    x_valid = 0 <= x < block_size
    y_valid = 0 <= y < block_size
    z_valid = 0 <= z < block_size
    
    if not (x_valid and y_valid and z_valid):
      out_of_bounds_count += 1
      continue
    
    # Skip if block_id is padding (0)
    if block_id == pad_token_id or block_id == 0:
      continue
    
    # Check for collision (overwriting existing block)
    if schematic[y, z, x, 0] > 0:
      collision_count += 1
    
    # Place block in schematic (format is y, z, x, channels)
    # Channel 0: block ID
    # Channel 1: metadata (set to 0 for now since we don't have that info)
    schematic[y, z, x, 0] = block_id
    schematic[y, z, x, 1] = 0
    placed_count += 1
  
  # Debug: print statistics if there are issues
  if len(block_ids) > 0:
    if out_of_bounds_count > 0 or collision_count > 0 or placed_count < len(block_ids) * 0.9:
      print(f'[Schematic conversion] Placed {placed_count}/{len(block_ids)} blocks, '
            f'{out_of_bounds_count} out-of-bounds, {collision_count} collisions, '
            f'coords range: [{min_coord:.1f}, {max_coord:.1f}], needs_shift: {needs_shift}')
  
  return schematic


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  
  # Check if this is Craft3D data (doesn't have gen_ppl_metric)
  is_craft3d = getattr(config.data, "type", None) == "craft3d"
  
  # For Craft3D, get coordinates and ground truth blocks from dataset
  # Default to validation set, but allow override via config
  # Or load from a coordinate file if specified
  use_train_set = getattr(config.eval, 'sample_from_train', False)
  coordinate_file = getattr(config.eval, 'coordinate_file', None)
  coords = None
  ground_truth_blocks = None
  attention_mask = None
  pad_token_id = None
  if is_craft3d:
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    
    # Check if we should load coordinates from a file
    if coordinate_file is not None:
      logger.info(f'Loading coordinates from file: {coordinate_file}')
      file_coords, file_block_ids = load_coords_from_schematic(coordinate_file, block_size=32)
      
      if len(file_coords) == 0:
        raise ValueError(f'No occupied blocks found in coordinate file: {coordinate_file}')
      
      logger.info(f'Loaded {len(file_coords)} coordinates from file')
      logger.info(f'Coordinate range: X=[{file_coords[:, 0].min():.1f}, {file_coords[:, 0].max():.1f}], '
                  f'Y=[{file_coords[:, 1].min():.1f}, {file_coords[:, 1].max():.1f}], '
                  f'Z=[{file_coords[:, 2].min():.1f}, {file_coords[:, 2].max():.1f}]')
      
      # Convert to tensors and prepare for sampling
      seq_len = config.model.length
      num_coords = len(file_coords)
      
      # Create coordinates tensor with padding if needed
      coords_tensor = torch.zeros((1, seq_len, 3), dtype=torch.float32)
      coords_tensor[0, :num_coords] = torch.from_numpy(file_coords)
      
      # Create attention mask (1 for real coordinates, 0 for padding)
      attention_mask_tensor = torch.zeros((1, seq_len), dtype=torch.long)
      attention_mask_tensor[0, :num_coords] = 1
      
      coords = coords_tensor
      attention_mask = attention_mask_tensor
      ground_truth_blocks = None  # No ground truth when using coordinate file
      
      # Adjust batch size to 1 when using coordinate file
      eval_batch_size = 1
      logger.info(f'Using coordinate file, setting batch size to 1')
    else:
      # Load from dataset as before
      dataset_name = 'training' if use_train_set else 'validation'
    logger.info(f'Loading coordinates from {dataset_name} dataset...')
    import dataloader
    if use_train_set:
      train_ds, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True, valid_seed=config.seed)
      source_ds = train_ds
    else:
      _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
      source_ds = valid_ds
    # Get a batch from the dataloader
    import random
    batch_index = getattr(config.eval, 'batch_index', None)
    sample_index_in_batch = getattr(config.eval, 'sample_index_in_batch', None)
    
    ds_iter = iter(source_ds)
    num_batches = len(source_ds)
    
    if batch_index is not None:
      # Use specific batch index
      if batch_index >= num_batches:
        logger.warning(f'batch_index {batch_index} is >= number of batches ({num_batches}), using batch 0 instead')
        batch_index = 0
      logger.info(f'Using batch index: {batch_index} (total batches: {num_batches})')
      for _ in range(batch_index):
        try:
          next(ds_iter)
        except StopIteration:
          ds_iter = iter(source_ds)
          break
    else:
      # Random batch selection (default behavior)
      num_skip = random.randint(0, min(10, num_batches - 1))
      logger.info(f'Randomly selecting batch (skipping {num_skip} batches)')
      for _ in range(num_skip):
        try:
          next(ds_iter)
        except StopIteration:
          ds_iter = iter(source_ds)
          break
    
    batch = next(ds_iter)
    coords = batch['coords']  # Shape: (batch_size, seq_len, 3)
    ground_truth_blocks = batch['input_ids']  # Shape: (batch_size, seq_len) - ground truth block IDs
    attention_mask = batch['attention_mask']  # Shape: (batch_size, seq_len) - 1 for real tokens, 0 for padding
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    
    # Select specific sample index if provided
    if sample_index_in_batch is not None:
      batch_size = coords.shape[0]
      if sample_index_in_batch >= batch_size:
        logger.warning(f'sample_index_in_batch {sample_index_in_batch} is >= batch size ({batch_size}), using index 0 instead')
        sample_index_in_batch = 0
      logger.info(f'Using sample index {sample_index_in_batch} from batch (batch size: {batch_size})')
      coords = coords[sample_index_in_batch:sample_index_in_batch+1]  # Keep batch dimension
      ground_truth_blocks = ground_truth_blocks[sample_index_in_batch:sample_index_in_batch+1]
      attention_mask = attention_mask[sample_index_in_batch:sample_index_in_batch+1] if attention_mask is not None else None
    
    # Ensure coordinates match eval_batch_size for sampling (if not using specific sample index)
    eval_batch_size = config.loader.eval_batch_size
    if sample_index_in_batch is None and coords.shape[0] != eval_batch_size:
      logger.info(f'Adjusting batch size from {coords.shape[0]} to {eval_batch_size} for sampling')
      coords = coords[:eval_batch_size]
      ground_truth_blocks = ground_truth_blocks[:eval_batch_size]
      attention_mask = attention_mask[:eval_batch_size] if attention_mask is not None else None
    
    logger.info(f'Loaded coordinates shape: {coords.shape}')
    logger.info(f'Loaded ground truth blocks shape: {ground_truth_blocks.shape}')
    logger.info(f'Loaded attention mask shape: {attention_mask.shape}')
  
  if not is_craft3d and hasattr(model, 'gen_ppl_metric'):
    model.gen_ppl_metric.reset()
    
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  
  samples = None
  text_samples = None
  
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      if is_craft3d and coords is not None:
        # Pass coordinates directly to sampling
        samples = model.restore_model_and_sample(
          num_steps=config.sampling.steps, coords=coords.to(model.device))
      else:
        samples = model.restore_model_and_sample(
          num_steps=config.sampling.steps)
      if is_craft3d:
        # For Craft3D, samples are block IDs (not text)
        # Mask out padding positions - set them back to pad_token_id
        if attention_mask is not None and pad_token_id is not None:
          # attention_mask: 1 for real tokens, 0 for padding
          # Set padding positions (where attention_mask == 0) back to pad_token_id
          padding_mask = (attention_mask == 0).to(samples.device)
          samples = torch.where(padding_mask, 
                                torch.tensor(pad_token_id, device=samples.device, dtype=samples.dtype),
                                samples)
        logger.info(f'Generated samples shape: {samples.shape}')
        logger.info(f'Sample statistics: min={samples.min()}, max={samples.max()}, unique={len(torch.unique(samples))}')
        text_samples = samples  # Return block IDs directly
      else:
        text_samples = model.tokenizer.batch_decode(samples)
        if hasattr(model, 'compute_generative_perplexity'):
          model.compute_generative_perplexity(text_samples)
  
  if is_craft3d:
    if samples is not None:
      print('\n' + '='*70)
      print('SAMPLING RESULTS')
      print('='*70)
      print(f'Generated samples shape: {samples.shape}')
      print(f'Generated sample statistics: min={samples.min()}, max={samples.max()}, unique={len(torch.unique(samples))}')
      
      if ground_truth_blocks is not None:
        print(f'\nGround truth blocks shape: {ground_truth_blocks.shape}')
        print(f'Ground truth statistics: min={ground_truth_blocks.min()}, max={ground_truth_blocks.max()}, unique={len(torch.unique(ground_truth_blocks))}')
        
        # Print comparison for first sample (excluding padding)
        if samples.shape[0] > 0 and ground_truth_blocks.shape[0] > 0:
          seq_len = min(samples.shape[1], ground_truth_blocks.shape[1])
          # Move to same device for comparison
          samples_cpu = samples[0, :seq_len].cpu()
          gt_cpu = ground_truth_blocks[0, :seq_len].cpu()
          
          # Only compare non-padding positions
          if attention_mask is not None:
            attn_mask_cpu = attention_mask[0, :seq_len].cpu()
            # Filter to only non-padding positions (where attention_mask == 1)
            non_padding_mask = (attn_mask_cpu == 1)
            if non_padding_mask.sum() > 0:
              samples_non_pad = samples_cpu[non_padding_mask]
              gt_non_pad = gt_cpu[non_padding_mask]
              matches = (samples_non_pad == gt_non_pad).sum().item()
              total = non_padding_mask.sum().item()
              accuracy = matches / total * 100 if total > 0 else 0
              
              print(f'\nComparison (first sample, non-padding positions only):')
              print(f'  Non-padding positions: {total}/{seq_len}')
              print(f'  Matching positions: {matches}/{total} ({accuracy:.2f}%)')
              
              # Get coordinates for non-padding positions
              coords_cpu = coords[0, :seq_len].cpu() if coords is not None else None
              coords_non_pad = coords_cpu[non_padding_mask] if coords_cpu is not None else None
              
              print(f'\nGenerated block IDs (first sample, all non-padding positions):')
              print(samples_non_pad.tolist())
              
              print(f'\nGround truth block IDs (first sample, all non-padding positions):')
              print(gt_non_pad.tolist())
              
              if coords_non_pad is not None:
                print(f'\nCoordinates (first sample, all non-padding positions):')
                print(coords_non_pad.tolist())
              
              # Print positions where they match vs mismatch (with original sequence position indices and coordinates)
              match_positions = (samples_non_pad == gt_non_pad)
              original_positions = torch.nonzero(non_padding_mask, as_tuple=False).squeeze(1).cpu()
              print(f'\nPosition-by-position comparison (showing original sequence positions):')
              for filtered_idx, (gen_id, gt_id, matches_pos) in enumerate(zip(samples_non_pad.tolist(), gt_non_pad.tolist(), match_positions.tolist())):
                orig_pos = original_positions[filtered_idx].item()
                match_symbol = '✓' if matches_pos else '✗'
                coord_str = ''
                if coords_non_pad is not None:
                  coord = coords_non_pad[filtered_idx].tolist()
                  coord_str = f', Coords=({coord[0]:.1f},{coord[1]:.1f},{coord[2]:.1f})'
                print(f'  Seq Pos {orig_pos:4d}: Generated={gen_id:3d}, Ground Truth={gt_id:3d} {match_symbol}{coord_str}')
            else:
              print(f'\nComparison: No non-padding positions found')
          else:
            # Fallback: compare all positions if no attention_mask
            matches = (samples_cpu == gt_cpu).sum().item()
            total = seq_len
            accuracy = matches / total * 100 if total > 0 else 0
            print(f'\nComparison (first sample, all positions):')
            print(f'  Matching positions: {matches}/{total} ({accuracy:.2f}%)')
            print(f'\nGenerated block IDs (first sample, all positions):')
            print(samples_cpu.tolist())
            print(f'\nGround truth block IDs (first sample, all positions):')
            print(gt_cpu.tolist())
      else:
        # If no ground truth, just print all generated positions
        print(f'\nGenerated block IDs (first sample, all positions):')
        print(samples[0].cpu().tolist() if samples.shape[0] > 0 else samples.cpu().tolist())
      print('='*70 + '\n')
      
      # Save schematics for all samples
      if samples is not None and coords is not None:
        # Create output directory for schematics
        output_dir = Path(os.getcwd()) / 'schematics'
        output_dir.mkdir(exist_ok=True)
        logger.info(f'Saving schematics to {output_dir}')
        
        batch_size = samples.shape[0]
        for sample_idx in range(batch_size):
          # Save generated schematic
          gen_schematic = _blocks_to_schematic(
            block_ids=samples[sample_idx],
            coords=coords[sample_idx],
            attention_mask=attention_mask[sample_idx] if attention_mask is not None else None,
            pad_token_id=pad_token_id,
            block_size=32
          )
          gen_path = output_dir / f'sample_{sample_idx:04d}_generated.npy'
          np.save(gen_path, gen_schematic)
          
          # Save ground truth schematic if available
          if ground_truth_blocks is not None:
            gt_schematic = _blocks_to_schematic(
              block_ids=ground_truth_blocks[sample_idx],
              coords=coords[sample_idx],
              attention_mask=attention_mask[sample_idx] if attention_mask is not None else None,
              pad_token_id=pad_token_id,
              block_size=32
            )
            gt_path = output_dir / f'sample_{sample_idx:04d}_ground_truth.npy'
            np.save(gt_path, gt_schematic)
        
        logger.info(f'Saved {batch_size} generated schematics and {batch_size if ground_truth_blocks is not None else 0} ground truth schematics')
  else:
    print('Text samples:', text_samples)
    if not config.sampling.semi_ar and hasattr(model, 'gen_ppl_metric'):
      print('Generative perplexity:',
            model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()
  
  # [(x,y,z,c) x n ]
  