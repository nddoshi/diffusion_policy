"""
Image-based dataset for the Box Touch environment.

Loads a zarr ReplayBuffer containing:
  - obs: (T, 23) float32 — full state (used for agent_pos extraction)
  - action: (T, 7) float32 — target joint positions
  - img: (T, H, W, 3) uint8 — camera images

Observation dict for training:
  - 'image': (T, 3, H, W) float32 in [0, 1]
  - 'agent_pos': (T, 7) float32 — joint positions (first 7 dims of obs)

Following the pattern of PushTImageDataset.
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class BoxTouchImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'obs', 'action'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            # agent_pos is the first 7 dims of obs (joint positions)
            'agent_pos': self.replay_buffer['obs'][..., :7],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Extract joint positions (first 7 dims) as agent_pos
        agent_pos = sample['obs'][:, :7].astype(np.float32)  # (T, 7)

        # Convert image: (T, H, W, 3) uint8 → (T, 3, H, W) float32 [0, 1]
        image = np.moveaxis(sample['img'], -1, 1) / 255.0

        data = {
            'obs': {
                'image': image.astype(np.float32),   # T, 3, H, W
                'agent_pos': agent_pos,               # T, 7
            },
            'action': sample['action'].astype(np.float32),  # T, 7
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
