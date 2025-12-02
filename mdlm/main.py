import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

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
  
  # For Craft3D, get coordinates and ground truth blocks from validation dataset
  coords = None
  ground_truth_blocks = None
  attention_mask = None
  pad_token_id = None
  if is_craft3d:
    logger.info('Loading coordinates from validation dataset...')
    import dataloader
    _, valid_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=config.seed)
    # Get a random batch from validation dataloader
    import random
    val_iter = iter(valid_ds)
    # Skip a random number of batches for randomness
    num_skip = random.randint(0, min(10, len(valid_ds) - 1))
    for _ in range(num_skip):
      try:
        next(val_iter)
      except StopIteration:
        val_iter = iter(valid_ds)
        break
    batch = next(val_iter)
    coords = batch['coords']  # Shape: (batch_size, seq_len, 3)
    ground_truth_blocks = batch['input_ids']  # Shape: (batch_size, seq_len) - ground truth block IDs
    attention_mask = batch['attention_mask']  # Shape: (batch_size, seq_len) - 1 for real tokens, 0 for padding
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
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
      print(f'\nGenerated block IDs (first sample, first 100 positions):')
      print(samples[0, :100].cpu().tolist() if samples.shape[0] > 0 else samples)
      
      if ground_truth_blocks is not None:
        print(f'\nGround truth blocks shape: {ground_truth_blocks.shape}')
        print(f'Ground truth statistics: min={ground_truth_blocks.min()}, max={ground_truth_blocks.max()}, unique={len(torch.unique(ground_truth_blocks))}')
        print(f'\nGround truth block IDs (first sample, first 100 positions):')
        print(ground_truth_blocks[0, :100].cpu().tolist() if ground_truth_blocks.shape[0] > 0 else ground_truth_blocks)
        
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
            else:
              print(f'\nComparison: No non-padding positions found')
          else:
            # Fallback: compare all positions if no attention_mask
            matches = (samples_cpu == gt_cpu).sum().item()
            total = seq_len
            accuracy = matches / total * 100 if total > 0 else 0
            print(f'\nComparison (first sample, all positions):')
            print(f'  Matching positions: {matches}/{total} ({accuracy:.2f}%)')
      print('='*70 + '\n')
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
  