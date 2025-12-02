import math
import os
import typing
import torch
import transformers

import utils
from helper.craft3d_dataset import Craft3DDataset

LOGGER = utils.get_logger(__name__)


def get_tokenizer(config):
  """Get tokenizer for Craft3D data.
  
  For Craft3D, the tokenizer is mainly used to provide vocab_size and special token IDs.
  Block IDs are used directly as token IDs without actual tokenization.
  """
  # For Craft3D, we can use a simple tokenizer or create a minimal one
  # The tokenizer mainly needs to provide vocab_size and special token IDs
  if hasattr(config.data, 'tokenizer_name_or_path'):
    try:
      tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.data.tokenizer_name_or_path)
    except:
      # Fallback: create a minimal tokenizer-like object
      class MinimalTokenizer:
        def __init__(self, vocab_size, pad_token_id, bos_token_id, eos_token_id):
          self.vocab_size = vocab_size
          self.pad_token_id = pad_token_id
          self.bos_token_id = bos_token_id
          self.eos_token_id = eos_token_id
          self.pad_token = '[PAD]'
          self.bos_token = '[BOS]'
          self.eos_token = '[EOS]'
          if hasattr(config.data, 'mask_token_id'):
            self.mask_token_id = config.data.mask_token_id
            self.mask_token = '[MASK]'
          else:
            self.mask_token_id = None
            self.mask_token = None
      
      tokenizer = MinimalTokenizer(
        vocab_size=getattr(config.data, 'vocab_size', 2048),
        pad_token_id=getattr(config.data, 'pad_token_id', 0),
        bos_token_id=getattr(config.data, 'bos_token_id', 0),
        eos_token_id=getattr(config.data, 'eos_token_id', 0)
      )
  else:
    # Use config values directly
    class MinimalTokenizer:
      def __init__(self, vocab_size, pad_token_id, bos_token_id, eos_token_id, mask_token_id=None):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token = '[PAD]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        if mask_token_id is not None:
          self.mask_token_id = mask_token_id
          self.mask_token = '[MASK]'
        else:
          self.mask_token_id = None
          self.mask_token = None
    
    tokenizer = MinimalTokenizer(
      vocab_size=getattr(config.data, 'vocab_size', 2048),
      pad_token_id=getattr(config.data, 'pad_token_id', 0),
      bos_token_id=getattr(config.data, 'bos_token_id', 0),
      eos_token_id=getattr(config.data, 'eos_token_id', 0),
      mask_token_id=getattr(config.data, 'mask_token_id', None)
    )

  return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  """Get dataloaders for Craft3D dataset."""
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.loader.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size {config.loader.eval_batch_size} '
      f'not divisible by {num_gpus}.')
  
  data_type = getattr(config.data, "type", "craft3d")
  
  if data_type != "craft3d":
    raise ValueError(f"Only 'craft3d' data type is supported. Got: {data_type}")

  if skip_train:
    train_set = None
  else:
    train_set = Craft3DSequenceDataset(
      data_dir=config.data.craft3d_dir,
      split=getattr(config.data, "train_split", "train"),
      seq_len=config.model.length,
      pad_token_id=_get_pad_token_id(tokenizer),
      max_samples=getattr(config.data, "max_train_samples", None),
    )
  
  if skip_valid:
    valid_set = None
  else:
    valid_set = Craft3DSequenceDataset(
      data_dir=config.data.craft3d_dir,
      split=getattr(config.data, "valid_split", "val"),
      seq_len=config.model.length,
      pad_token_id=_get_pad_token_id(tokenizer),
      max_samples=getattr(config.data, "max_valid_samples", None),
    )

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=True,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    # Will be used in generative perplexity calculation
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


def _get_pad_token_id(tokenizer):
  """Get pad token ID from tokenizer or config."""
  if tokenizer is None:
    return 0
  if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
    return tokenizer.pad_token_id
  if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
    return tokenizer.eos_token_id
  return 0


class Craft3DSequenceDataset(torch.utils.data.Dataset):
  """Dataset for Craft3D sequences with padding."""
  
  def __init__(self, data_dir, split, seq_len, pad_token_id=0, max_samples=None):
    self.dataset = Craft3DDataset(
      data_dir=data_dir,
      subset=split,
      max_samples=max_samples,
    )
    self.seq_len = seq_len
    self.pad_token_id = pad_token_id

  def __len__(self):
    return self.dataset.get_num_houses()

  def __getitem__(self, idx):
    annotation = self.dataset.get_house(idx)
    block_types = annotation[:, 0]
    coords = annotation[:, 1:].float()

    length = min(len(block_types), self.seq_len)
    tokens = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
    coords_tensor = torch.zeros((self.seq_len, 3), dtype=torch.float32)
    attn_mask = torch.zeros((self.seq_len,), dtype=torch.long)

    if length > 0:
      tokens[:length] = block_types[:length]
      coords_tensor[:length] = coords[:length]
      attn_mask[:length] = 1

    return {
      "input_ids": tokens,
      "attention_mask": attn_mask,
      "coords": coords_tensor,
    }


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
