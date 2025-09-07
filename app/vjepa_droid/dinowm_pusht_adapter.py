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
    frameskip=1,
):
    dataset = DinoWMVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=frameskip,
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

    logger.info("DinoWM VideoDataset data loader created")

    return data_loader, dist_sampler


class DinoWMVideoDataset(torch.utils.data.Dataset):
    """DinoWM dataset adapter for VJEPA2 interface."""

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
        
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load DinoWM data files
        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "rel_actions.pth").float()
        
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # Load shapes if available
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                self.shapes = pickle.load(f)
        else:
            self.shapes = ['T'] * len(self.states)

        # Limit number of rollouts if specified
        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
            self.states = self.states[:n]
            self.actions = self.actions[:n]
            self.seq_lengths = self.seq_lengths[:n]
            self.shapes = self.shapes[:n]

        print(f"Loaded {len(self.states)} rollouts from DinoWM format")
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")

    def __getitem__(self, index):
        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(index)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {index=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())

        # # Print shapes of all outputs in __getitem__
        # print("\nShapes in DinoWMVideoDataset __getitem__:")
        # print(f"buffer shape: {buffer.shape if hasattr(buffer, 'shape') else 'no shape'}")
        # print(f"actions shape: {actions.shape if hasattr(actions, 'shape') else 'no shape'}")
        # print(f"states shape: {states.shape if hasattr(states, 'shape') else 'no shape'}")
        # print(f"extrinsics shape: {extrinsics.shape if hasattr(extrinsics, 'shape') else 'no shape'}")
        # print(f"indices shape: {indices.shape if hasattr(indices, 'shape') else 'no shape'}")

        return buffer, actions, states, extrinsics, indices

    def poses_to_diffs_2d(self, poses):
        """Compute action diffs from poses - only use first 2 dimensions (XY movement)."""
        # poses: [T, D] where D >= 2
        diffs = poses[1:] - poses[:-1]  # [T-1, D]
        # Zero out all dimensions except first 2 to focus on XY movement
        if diffs.shape[1] > 2:
            diffs[:, 2:] = 0
        return diffs

    def loadvideo_decord(self, index):
        debug = False  # Set to False to disable debugging

        # Get sequence length for this episode
        seq_length = self.seq_lengths[index]
        
        # Load video
        vid_dir = self.data_path / "obses"
        video_path = vid_dir / f"episode_{index:03d}.mp4"
        
        if not video_path.exists():
            raise Exception(f"Video file not found: {video_path}")
            
        vr = VideoReader(str(video_path), num_threads=-1, ctx=cpu(0))
        vlen = len(vr)
        
        # Use fixed horizon (frames_per_clip) for consistent batching
        safe_frameskip = max(1, self.frameskip)
        frames_needed = self.frames_per_clip * safe_frameskip
        
        # Check if episode is long enough
        if seq_length < frames_needed:
            raise Exception(f"Episode {index} too short: seq_length={seq_length}, needed={frames_needed}")
        
        # Sample a window of frames_per_clip frames with frameskip
        # Random sampling like DROID, but constrained to available sequence length
        max_start = seq_length - frames_needed
        start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # Create indices with frameskip
        indices = np.arange(start_frame, start_frame + frames_needed, safe_frameskip).astype(np.int64)
        
        # Ensure indices don't exceed available video or state data
        indices = indices[indices < min(seq_length, vlen)]
        
        if len(indices) < self.frames_per_clip:
            raise Exception(f"Not enough valid indices for episode {index}: got {len(indices)}, need {self.frames_per_clip}")
        
        # Truncate to exactly frames_per_clip
        indices = indices[:self.frames_per_clip]
        
        # Get states and actions for this episode
        episode_states = self.states[index, :seq_length].numpy()  # [T, D]
        episode_actions = self.actions[index, :seq_length-1].numpy() if seq_length > 1 else np.zeros((0, self.actions.shape[-1]))  # [T-1, D]
        
        # Sample states at the exact indices
        states = episode_states[indices]  # [frames_per_clip, D]
        
        # For actions, sample at indices but handle T-1 dimension
        # Actions correspond to transitions between states, so we need indices[:-1] for actions
        if len(indices) > 1:
            action_indices = indices[:-1]  # Remove last index since actions are T-1
            action_indices = action_indices[action_indices < len(episode_actions)]
            if len(action_indices) > 0:
                actions = episode_actions[action_indices]  # [frames_per_clip-1, D]
            else:
                # Fallback: compute actions from states
                actions = self.poses_to_diffs_2d(states)
        else:
            # Single frame, no actions
            actions = np.zeros((0, states.shape[1]))
        
        # Zero out all but first 2 dimensions of actions (focus on XY movement)
        if actions.shape[0] > 0 and actions.shape[1] > 2:
            actions[:, 2:] = 0
        
        # Load video frames
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [frames_per_clip, H, W, C]
        
        # Keep video buffer as uint8 [0-255] - EXACTLY like DROID
        # Don't convert to float32 here - let transforms handle it
        
        # Keep video in THWC format (like DROID loader) - transforms handle the conversion
        
        # Pad states to 7 dimensions (VJEPA expects 7D states)
        if states.shape[1] < 7:
            padded_states = np.zeros((states.shape[0], 7), dtype=np.float64)  # Use float64 like DROID
            padded_states[:, :states.shape[1]] = states.astype(np.float64)
            states = padded_states
        else:
            states = states.astype(np.float64)
        
        # Ensure actions have same dimensionality as states (7D)
        if actions.shape[0] > 0:
            if actions.shape[1] < 7:
                padded_actions = np.zeros((actions.shape[0], 7), dtype=np.float64)  # Use float64 like DROID
                padded_actions[:, :min(actions.shape[1], 2)] = actions[:, :min(actions.shape[1], 2)]  # Keep only first 2 dims
                actions = padded_actions
            else:
                actions = actions.astype(np.float64)
                # Zero out all but first 2 dimensions
                if actions.shape[1] > 2:
                    actions[:, 2:] = 0
        else:
            # No actions case
            actions = actions.astype(np.float64)
        
        # Create dummy extrinsics - DROID uses (frames_per_clip, 6) shape
        extrinsics = np.zeros((self.frames_per_clip, 6), dtype=np.float64)  # 6D to match DROID exactly
        
        if self.transform is not None:
            buffer = self.transform(buffer)

        # Debug prints for shapes
        if debug:
            print(f"Episode {index} - Debug shapes:")
            print(f"  buffer shape: {buffer.shape}, dtype: {buffer.dtype}")
            print(f"  actions shape: {actions.shape}, dtype: {actions.dtype}")
            print(f"  states shape: {states.shape}, dtype: {states.dtype}")
            print(f"  extrinsics shape: {extrinsics.shape}, dtype: {extrinsics.dtype}")
            print(f"  indices shape: {indices.shape}, dtype: {indices.dtype}")
            print(f"  seq_length: {seq_length}")
            print(f"  frames_per_clip: {self.frames_per_clip}")
            print(f"  frameskip: {self.frameskip}")
            print(f"  actual_frames_loaded: {len(indices)}")
            print(f"  Note: actions is T-1 length (expected) - actions represent transitions between consecutive states")
            
            # Render rollout with agent position overlay
            self.render_rollout_debug(buffer, states, index)
            # intentionally terminate our script to avoid over saving videos
            sys.exit(0)
        
        return buffer, actions, states, extrinsics, indices

    def render_rollout_debug(self, buffer, states, episode_idx):
        """Debug function to render video with agent position overlay."""
        try:
            # Convert buffer from torch tensor to numpy if needed
            if torch.is_tensor(buffer):
                video_frames = buffer.cpu().numpy()
            else:
                video_frames = buffer
            
            # Buffer shape: [C, T, H, W] -> need [T, H, W, C] for cv2
            if video_frames.shape[0] == 3:  # Channels first
                video_frames = np.transpose(video_frames, (1, 2, 3, 0))  # [T, H, W, C]
            
            # Convert from float [0,1] to uint8 [0,255] if needed
            if video_frames.dtype == np.float32:
                video_frames = (video_frames * 255).astype(np.uint8)
            
            # Convert RGB to BGR for cv2
            video_frames = video_frames[..., ::-1]  # RGB -> BGR
            
            # Get agent positions (first 2 dimensions of state)
            agent_positions = states[:, :2]  # [T, 2]
            
            # Normalize positions to image coordinates
            # Assuming state positions are in some world coordinates, 
            # we need to map them to pixel coordinates [0, 255]
            pos_min = agent_positions.min(axis=0)
            pos_max = agent_positions.max(axis=0)
            pos_range = pos_max - pos_min
            
            # Avoid division by zero
            pos_range = np.where(pos_range == 0, 1, pos_range)
            
            # Map to image coordinates with some padding
            img_size = video_frames.shape[1:3]  # [H, W]
            padding = 20
            normalized_pos = (agent_positions - pos_min) / pos_range
            pixel_pos = (normalized_pos * (np.array(img_size)[::-1] - 2*padding) + padding).astype(int)  # [W, H] order for cv2
            
            # Create output directory in current working directory
            debug_dir = Path("./debug_videos")
            debug_dir.mkdir(exist_ok=True)
            
            # Create video writer
            output_path = debug_dir / f"episode_{episode_idx}_frameskip_{self.frameskip}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 5.0  # Reasonable fps for debug viewing
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                   (video_frames.shape[2], video_frames.shape[1]))  # (W, H)
            
            print(f"  Rendering debug video to: {output_path}")
            print(f"  Agent positions range: x=[{pos_min[0]:.3f}, {pos_max[0]:.3f}], y=[{pos_min[1]:.3f}, {pos_max[1]:.3f}]")
            
            # Render each frame with agent position
            for t in range(len(video_frames)):
                frame = video_frames[t].copy()
                
                # Draw agent position as green circle
                pos = pixel_pos[t]
                cv2.circle(frame, (pos[0], pos[1]), radius=5, color=(0, 255, 0), thickness=-1)
                
                # Draw trajectory up to current frame
                if t > 0:
                    for i in range(t):
                        prev_pos = pixel_pos[i]
                        curr_pos = pixel_pos[i+1] if i+1 <= t else pixel_pos[i]
                        cv2.line(frame, (prev_pos[0], prev_pos[1]), (curr_pos[0], curr_pos[1]), 
                               color=(0, 255, 255), thickness=1)  # Yellow trail
                
                # Add frame number and position text
                text = f"Frame {t}: pos=({agent_positions[t][0]:.2f}, {agent_positions[t][1]:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                writer.write(frame)
            
            writer.release()
            print(f"  Debug video saved successfully")
            
        except Exception as e:
            print(f"  Error rendering debug video: {e}")
            import traceback
            traceback.print_exc()

    def __len__(self):
        return len(self.seq_lengths)


class DinoWMDatasetAdapter:
    """Adapter to make DinoWM dataset work like PushT dataset for VJEPA2."""
    
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
        
        # Load data
        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "rel_actions.pth").float()
        
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # Load shapes if available
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                self.shapes = pickle.load(f)
        else:
            self.shapes = ['T'] * len(self.states)
        
        # Limit rollouts if specified
        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
        else:
            n = len(self.states)
            
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.shapes = self.shapes[:n]
        
        # Create proprios (first 2 dims of state for pusht-like interface)
        self.proprios = self.states[..., :2].clone()
        
        print(f"Loaded {n} rollouts from DinoWM format")
        
        # Set up normalization (dummy values for now)
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1] 
        self.proprio_dim = self.proprios.shape[-1]
        
        if normalize_action:
            # Use dummy normalization for now - can be computed later
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
            
            self.actions = (self.actions - self.action_mean) / self.action_std
            self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
    
    def get_seq_length(self, idx):
        return self.seq_lengths[idx]
    
    def get_frames(self, idx, frames):
        vid_dir = self.data_path / "obses"
        reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
        
        # Get data for requested frames
        act = self.actions[idx, frames]
        state = self.states[idx, frames] 
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]
        
        # Load video frames
        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        
        if self.transform:
            image = self.transform(image)
            
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}
    
    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self):
        return len(self.seq_lengths)
