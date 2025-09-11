# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
import pickle
import sys
from logging import getLogger
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from einops import rearrange

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
	data_path,
	batch_size,
	frames_per_clip=16,
	fps=5,
	crop_size=224,
	rank=0,
	world_size=1,
	camera_views=0,
	stereo_view=False,
	drop_last=True,
	num_workers=10,
	pin_mem=True,
	persistent_workers=True,
	collator=None,
	transform=None,
	camera_frame=False,
	tubelet_size=2,
	frameskip=1,  # Separate parameter for data subsampling
):
	dataset = DinoWMDeformableDataset(
		data_path=data_path,
		frames_per_clip=frames_per_clip,
		transform=transform,
		fps=fps,
		frameskip=frameskip,  # Use separate frameskip parameter
		camera_frame=camera_frame,
	)

	dist_sampler = torch.utils.data.distributed.DistributedSampler(
		dataset, num_replicas=world_size, rank=rank, shuffle=False #I changed this
	)

	data_loader = torch.utils.data.DataLoader(
		dataset,
		collate_fn=collator,
		sampler=dist_sampler,
		batch_size=batch_size,
		drop_last=drop_last,
		pin_memory=pin_mem,
		num_workers=num_workers,
		persistent_workers=(num_workers > 0) and persistent_workers,
	)

	logger.info("DinoWM Deformable Dataset data loader created")

	return data_loader, dist_sampler, dataset


class DinoWMDeformableDataset(torch.utils.data.Dataset):
	"""DinoWM deformable dataset adapter for VJEPA2 interface."""

	def __init__(
		self,
		data_path,
		frameskip=2,
		frames_per_clip=64,
		fps=5,
		transform=None,
		camera_frame=False,
		n_rollout=None,
	):
		self.data_path = Path(data_path)
		self.frames_per_clip = frames_per_clip
		self.frameskip = frameskip
		self.fps = fps
		self.transform = transform
		self.camera_frame = camera_frame

		# Load DinoWM deformable data files
		self.states = torch.load(self.data_path / "states.pth").float()
		# Rearrange states from (N, T, P, 4) to (N, T, P*4) like DinoWM
		self.states = rearrange(self.states, "N T P D -> N T (P D)")

		self.actions = torch.load(self.data_path / "actions.pth").float()

		# All sequences have same length in deformable dataset
		self.seq_lengths = [self.states.shape[1]] * len(self.states)

		# Create dummy proprios (1D)
		self.proprios = torch.zeros((len(self.states), self.states.shape[1], 1))

		# Limit number of rollouts if specified
		n_rollout=50
		if n_rollout is not None:
			n = min(n_rollout, len(self.states))
			self.states = self.states[:n]
			self.actions = self.actions[:n]
			self.seq_lengths = self.seq_lengths[:n]
			self.proprios = self.proprios[:n]

	def __getitem__(self, index):
		# -- keep trying to load data until you find a valid sample
		loaded_data = False
		while not loaded_data:
			try:
				buffer, actions, states, extrinsics, indices = self.load_deformable_data(index)
				loaded_data = True
			except Exception as e:
				loaded_data = False
				index = np.random.randint(self.__len__())

		return buffer, actions, states, extrinsics, indices

	def __len__(self):
		"""Return the number of sequences in the dataset."""

		# return len(self.states) #HRISH HARDCODED
		return 50

	def load_deformable_data(self, index):
		debug = False  # Set to False to disable debugging

		# Get sequence length for this episode
		seq_length = self.seq_lengths[index]

		# Load images from obses.pth
		obs_dir = self.data_path / f"{index:06d}"
		image_path = obs_dir / "obses.pth"

		if not image_path.exists():
			raise Exception(f"Obses file not found: {image_path}")

		# Load images: [T, H, W, C]
		images = torch.load(image_path).float()
		if debug:
			self.render_raw_images_debug(images, index)
			sys.exit(0)

		# Use fixed horizon (frames_per_clip) for consistent batching
		safe_frameskip = max(1, self.frameskip)
		frames_needed = self.frames_per_clip * safe_frameskip

		# Check if episode is long enough
		if seq_length < frames_needed:
			raise Exception(f"Episode {index} too short: seq_length={seq_length}, needed={frames_needed}")

		# # Sample a window of frames_per_clip frames with frameskip
		max_start = seq_length - frames_needed
		# start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
		start_frame = 0  # deterministic: always first window #HRISH HARDCODED


		# Create indices with frameskip
		indices = np.arange(start_frame, start_frame + frames_needed, safe_frameskip).astype(np.int64)

		# Ensure indices don't exceed available data
		indices = indices[indices < seq_length]

		if len(indices) < self.frames_per_clip:
			raise Exception(f"Not enough valid indices for episode {index}: got {len(indices)}, need {self.frames_per_clip}")

		# Truncate to exactly frames_per_clip
		indices = indices[:self.frames_per_clip]

		# Sample images at indices
		buffer = images[indices]  # [frames_per_clip, H, W, C]

		# Get states and actions for this episode
		episode_states = self.states[index, :seq_length].numpy()  # [T, state_dim]
		episode_actions = self.actions[index, :seq_length].numpy()  # [T, action_dim]

		# Create dummy states as a PyTorch tensor with shape [frames_per_clip, 7]

		states = torch.tensor(episode_states[indices], dtype=torch.float64)  # [frames_per_clip, state_dim]

		# Remove last action since actions at time t cause observations at time t+1
		actions = torch.tensor(episode_actions[indices[:-1]], dtype=torch.float64)  # [frames_per_clip-1, action_dim

		# Create dummy extrinsics as PyTorch tensor - DROID uses (frames_per_clip, 6) shape
		extrinsics = torch.zeros((self.frames_per_clip, 6), dtype=torch.float64)

		if self.transform is not None:
			buffer = self.transform(buffer)

		# # Print shapes of all outputs
		# print("buffer.shape: ", buffer.shape)
		# print("actions shape:", actions.shape)
		# print("states shape:", states.shape)
		# print("extrinsics shape:", extrinsics.shape)
		# print("indices shape:", indices.shape)
		
		return buffer, actions, states, extrinsics, indices