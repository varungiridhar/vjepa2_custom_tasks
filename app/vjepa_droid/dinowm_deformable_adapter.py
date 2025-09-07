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
		dataset, num_replicas=world_size, rank=rank, shuffle=True
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

	return data_loader, dist_sampler


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
		if n_rollout is not None:
			n = min(n_rollout, len(self.states))
			self.states = self.states[:n]
			self.actions = self.actions[:n]
			self.seq_lengths = self.seq_lengths[:n]
			self.proprios = self.proprios[:n]

		print(f"Loaded {len(self.states)} rollouts from DinoWM deformable format")
		print(f"States shape: {self.states.shape}")
		print(f"Actions shape: {self.actions.shape}")

	def __getitem__(self, index):
		# -- keep trying to load data until you find a valid sample
		loaded_data = False
		while not loaded_data:
			try:
				buffer, actions, states, extrinsics, indices = self.load_deformable_data(index)
				loaded_data = True
			except Exception as e:
				logger.info(f"Encountered exception when loading deformable data {index=} {e=}")
				loaded_data = False
				index = np.random.randint(self.__len__())

		return buffer, actions, states, extrinsics, indices

	def __len__(self):
		"""Return the number of sequences in the dataset."""
		return len(self.states)

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

		# Sample a window of frames_per_clip frames with frameskip
		max_start = seq_length - frames_needed
		start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0

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

		# # Convert to float32 [0,1] and rearrange to [T, C, H, W]
		# buffer = buffer / 255.0
		# buffer = rearrange(buffer, "T H W C -> T C H W")
		# print("buffer.shape: ", buffer.shape)

		# Get states and actions for this episode
		episode_states = self.states[index, :seq_length].numpy()  # [T, state_dim]
		episode_actions = self.actions[index, :seq_length].numpy()  # [T, action_dim]

		# Create dummy states as a PyTorch tensor with shape [frames_per_clip, 7]
		states = torch.zeros((self.frames_per_clip, 7), dtype=torch.float64)
		
		# Sample actions and convert to PyTorch tensor
		# Remove last action since actions at time t cause observations at time t+1
		actions = torch.tensor(episode_actions[indices[:-1]], dtype=torch.float64)  # [frames_per_clip-1, action_dim]

		# state is not really used in this case
		# # Pad states to 7 dimensions (VJEPA expects 7D states) - using first 7 dims or pad
		# if states.shape[1] < 7:
		# 	padded_states = np.zeros((states.shape[0], 7), dtype=np.float64)
		# 	padded_states[:, :states.shape[1]] = states.astype(np.float64)
		# 	states = padded_states
		# else:
		# 	states = states[:, :7].astype(np.float64)  # Take first 7 dimensions

		# Pad the actions to dim 7 (right now action dim is 4:
		# The action dimension (4 numbers) defines a pair of 2D points on the table, denoting a pushing action starting from one point to the other)
		# Actions shape should be [frames_per_clip-1, 7] since each action leads to the next observation
		actions = torch.nn.functional.pad(actions, (0, 3), mode='constant', value=0)

		# Create dummy extrinsics as PyTorch tensor - DROID uses (frames_per_clip, 6) shape
		extrinsics = torch.zeros((self.frames_per_clip, 6), dtype=torch.float64)

		if self.transform is not None:
			buffer = self.transform(buffer)

		# # Print shapes of all outputs
		# print("actions shape:", actions.shape)
		# print("states shape:", states.shape)
		# print("extrinsics shape:", extrinsics.shape)
		# print("indices shape:", indices.shape)
		
		return buffer, actions, states, extrinsics, indices

class DinoWMDeformableAdapter:
	"""Adapter to make DinoWM deformable dataset work like PushT dataset for VJEPA2."""

	def __init__(
		self,
		data_path: str,
		transform=None,
		n_rollout=None,
		normalize_action: bool = False,
		with_velocity: bool = False,
	):
		self.data_path = Path(data_path)
		self.transform = transform
		self.normalize_action = normalize_action
		self.with_velocity = with_velocity

		# Load data like DinoWM DeformDataset
		self.states = torch.load(self.data_path / "states.pth").float()
		self.states = rearrange(self.states, "N T P D -> N T (P D)")

		self.actions = torch.load(self.data_path / "actions.pth").float()

		self.seq_lengths = [self.states.shape[1]] * len(self.states)

		# Create proprios (dummy 1D)
		self.proprios = torch.zeros((len(self.states), self.states.shape[1], 1))

		# Limit rollouts if specified
		if n_rollout is not None:
			n = min(n_rollout, len(self.states))
		else:
			n = len(self.states)

		self.states = self.states[:n]
		self.actions = self.actions[:n]
		self.seq_lengths = self.seq_lengths[:n]
		self.proprios = self.proprios[:n]

		print(f"Loaded {n} rollouts from DinoWM deformable format")

		# Set up normalization
		self.action_dim = self.actions.shape[-1]
		self.state_dim = self.states.shape[-1]
		self.proprio_dim = self.proprios.shape[-1]

		if normalize_action:
			# Compute actual normalization statistics
			self.action_mean, self.action_std = self.get_data_mean_std(
				self.actions, self.seq_lengths
			)
			self.state_mean, self.state_std = self.get_data_mean_std(
				self.states, self.seq_lengths
			)
			self.proprio_mean, self.proprio_std = self.get_data_mean_std(
				self.proprios, self.seq_lengths
			)

			self.actions = (self.actions - self.action_mean) / self.action_std
			self.states = (self.states - self.state_mean) / self.state_std
			self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
		else:
			self.action_mean = torch.zeros(self.action_dim)
			self.action_std = torch.ones(self.action_dim)
			self.state_mean = torch.zeros(self.state_dim)
			self.state_std = torch.ones(self.state_dim)
			self.proprio_mean = torch.zeros(self.proprio_dim)
			self.proprio_std = torch.ones(self.proprio_dim)

	def get_data_mean_std(self, data, seq_lengths):
		all_data = []
		for traj in range(len(seq_lengths)):
			traj_len = seq_lengths[traj]
			traj_data = data[traj, :traj_len]
			all_data.append(traj_data)
		all_data = torch.vstack(all_data)
		data_mean = torch.mean(all_data, dim=0)
		data_std = torch.std(all_data, dim=0) + 1e-6
		return data_mean, data_std

	def get_seq_length(self, idx):
		return self.seq_lengths[idx]

	def get_frames(self, idx, frames):
		obs_dir = self.data_path / f"{idx:06d}"
		image = torch.load(obs_dir / "obses.pth")

		# Get data for requested frames
		act = self.actions[idx, frames]
		state = self.states[idx, frames]
		proprio = self.proprios[idx, frames]

		# Load video frames
		image = image[frames]  # THWC
		image = rearrange(image, "T H W C -> T C H W") / 255.0

		if self.transform:
			image = self.transform(image)

		obs = {"visual": image, "proprio": proprio}
		return obs, act, state, {}  # infos is empty dict

	def __getitem__(self, idx):
		return self.get_frames(idx, range(self.get_seq_length(idx)))

	def __len__(self):
		return len(self.seq_lengths)
