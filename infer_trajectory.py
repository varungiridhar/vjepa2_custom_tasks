import os
import random
import sys
import cv2
import numpy as np
from sklearn import base
import torch
# import gymnasium as gym
import gym
from pathlib import Path
from decord import VideoReader, cpu
from omegaconf import OmegaConf
import copy
from torch import default_generator, randperm
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
# Ensure we can import the Chamfer function from FlexEnvWrapper
from env.deformable_env.FlexEnvWrapper import chamfer_distance


sys.path.insert(0, "..")

from app.vjepa_droid.utils import init_video_model, load_pretrained, load_checkpoint
from app.vjepa_droid.dinowm_deformable_adapter import init_data
from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.world_model_wrapper import WorldModel
import env  # This registers the pusht environment with gymnasium
from datasets import traj_dset

def world_to_image_coords(world_pos, image_size=224, world_size=512):
	"""Convert world coordinates to image coordinates."""
	x, y = world_pos
	# Scale from world coordinates (0-512) to image coordinates (0-image_size)
	scale = image_size / world_size
	return (int(x * scale), int(y * scale))






# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total



def random_split_traj(
	dataset,
	lengths,
	generator=default_generator,):
      
	if sum(lengths) != len(dataset):  # type: ignore[arg-type]
		raise ValueError(
			"Sum of input lengths does not equal the length of the input dataset!"
		)

	indices = randperm(sum(lengths), generator=generator).tolist()
	print(
		[
			indices[offset - length : offset]
			for offset, length in zip(_accumulate(lengths), lengths)
		]
	)
	list_of_indices = [
		indices[offset - length : offset]
		for offset, length in zip(_accumulate(lengths), lengths)
	]


	dset = []
	for indices in list_of_indices:
		ret = []
		for i in indices:
			ret.append(dataset[i])
		dset.append(ret)

	return dset


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):

    dataset_length = len(dataset)
    
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_traj_segment_from_dset(dset, traj_len=6): # frameskip(1)*goal_h +1
	states = []
	actions = []
	observations = []
	env_info = []

	# Check if any trajectory is long enough


	# print("dset: ", len(self.dset))
	print("dset: ", len(dset))
      
	for k in dset[0]:
		print(k.shape)

	valid_traj = [
		dset[i][0].shape[0]
		for i in range(len(dset))
		if dset[i][0].shape[0] >= traj_len
	]


	if len(valid_traj) == 0:
		raise ValueError("No trajectory in the dataset is long enough.")

	# sample init_states from dset
	for i in range(1):
		max_offset = -1
		while max_offset < 0:  # filter out traj that are not long enough
			traj_id = 1
			# traj_id = random.randint(0, len(dset) - 1) 
			print("traj_id: ", traj_id)
			obs, act, state, e_info, _ = dset[traj_id]
			max_offset = obs.shape[0] - traj_len
		state = state.numpy()
		# offset = random.randint(0, max_offset)
		offset = 0 # HRISH HARDCODED

		state = state[offset : offset + traj_len]
		obs = obs[offset : offset + traj_len]
		act = act[offset : offset + traj_len]  # frameskip(1)*goal_h
		actions.append(act)
		states.append(state)
		observations.append(obs)
		env_info.append(e_info)
	return observations, states, actions, env_info



def draw_mpc_action_arrow(image, agent_pos, action, color=(0, 0, 255), thickness=2, arrow_length=30):
	"""Draw an arrow showing the MPC action direction."""
	# Convert positions to image coordinates
	agent_img = world_to_image_coords(agent_pos)
	
	# Action is [dx, dy, dz, dr, dp, dy, dgripper] - we only care about dx, dy
	action_dx, action_dy = action[:2]
	
	# Normalize action vector for arrow length
	action_magnitude = np.sqrt(action_dx**2 + action_dy**2)
	if action_magnitude > 0:
		# Scale arrow to fixed length for visibility
		scale_factor = arrow_length / action_magnitude
		arrow_dx = action_dx * scale_factor
		arrow_dy = action_dy * scale_factor
		
		arrow_end = (int(agent_img[0] + arrow_dx), int(agent_img[1] + arrow_dy))
		
		# Draw arrow
		cv2.arrowedLine(image, agent_img, arrow_end, color, thickness, tipLength=0.3)
	
	return image


def draw_predicted_pose(image, predicted_pos, color=(255, 0, 0), radius=8):
	"""Draw the predicted next pose of the agent."""
	# Convert to image coordinates
	pred_img = world_to_image_coords(predicted_pos)
	
	# Draw circle at predicted position
	cv2.circle(image, pred_img, radius, color, -1)  # Filled circle
	cv2.circle(image, pred_img, radius, (0, 0, 0), 2)  # Black outline
	
	return image


def draw_debug_legend(image, debug_mode):
	"""Draw a legend explaining the debug visualizations."""
	if not debug_mode:
		return image
	
	height, width = image.shape[:2]
	
	# Draw legend in top-left corner
	legend_x, legend_y = 10, 30
	line_height = 25
	
	# Background rectangle for legend
	cv2.rectangle(image, (legend_x-5, legend_y-20), (legend_x+200, legend_y+50), (255, 255, 255), -1)
	cv2.rectangle(image, (legend_x-5, legend_y-20), (legend_x+200, legend_y+50), (0, 0, 0), 1)
	
	# MPC Action arrow legend
	cv2.arrowedLine(image, (legend_x+10, legend_y), (legend_x+30, legend_y), (0, 0, 255), 2, tipLength=0.3)
	cv2.putText(image, "MPC Action", (legend_x+35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	
	# Predicted pose legend
	cv2.circle(image, (legend_x+20, legend_y+line_height), 4, (255, 0, 0), -1)
	cv2.circle(image, (legend_x+20, legend_y+line_height), 4, (0, 0, 0), 1)
	cv2.putText(image, "Predicted Pose", (legend_x+35, legend_y+line_height+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	
	return image



def save_side_by_side_video(
    frames,
    goal_frame: torch.Tensor,
    out_path: str,
    fps: int = 10,
    codec: str = "mp4v",
) -> None:
    """
    Save a video where the left side shows the sequence in `frames` and the right
    side shows the constant `goal_frame`.

    Args:
        frames: list/tuple of T tensors or a single tensor of shape
                - (T, C, H, W) or (T, H, W, C), or a Python list of tensors
                Each frame can be float in [0,1] or [0,255], or uint8. On CPU or CUDA.
        goal_frame: tensor of shape (C, H, W) or (H, W, C). Float or uint8. CPU or CUDA.
        out_path: output video filepath (e.g., "out.mp4").
        fps: frames per second for the video.
        codec: fourcc codec (default "mp4v" for .mp4).
    """
    def _to_hwc_rgb_uint8(t) -> np.ndarray:
        """Convert a torch.Tensor or np.ndarray to HWC RGB uint8.
        Accepts (C,H,W), (H,W,C), (H,W), or a leading batch dim (1,...).
        Handles floats in [0,1] or [0,255], and integer types.
        """
        import numpy as _np
        import torch as _torch

        # Convert torch -> numpy early
        if isinstance(t, _torch.Tensor):
            t = t.detach().to("cpu").numpy()

        if not isinstance(t, _np.ndarray):
            raise TypeError(f"Unsupported type for image: {type(t)}")

        # Remove batch if present
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]

        # Ensure HWC layout
        if t.ndim == 3:
            if t.shape[-1] in (1, 3, 4):
                pass  # already HWC
            elif t.shape[0] in (1, 3, 4):
                t = _np.transpose(t, (1, 2, 0))  # CHW -> HWC
        elif t.ndim == 2:
            t = t[..., None]  # HWC with 1 channel
        else:
            raise ValueError(f"Unsupported array shape: {t.shape}")

        # If single channel, repeat to 3
        if t.shape[-1] == 1:
            t = _np.repeat(t, 3, axis=-1)

        # Convert dtype to uint8, supporting [0,1] or [0,255]
        if _np.issubdtype(t.dtype, _np.floating):
            t_max = float(t.max()) if t.size > 0 else 1.0
            if t_max <= 1.0:
                t = _np.round(t * 255.0)
            t = _np.clip(t, 0, 255).astype(_np.uint8)
        else:
            t = _np.clip(t, 0, 255).astype(_np.uint8)

        return t

    # Normalize input frames to list of arrays (H,W,3) RGB uint8
    if isinstance(frames, torch.Tensor):
        if frames.ndim not in (4,):  # TCHW or THWC
            raise ValueError("`frames` tensor must be 4D (T, C, H, W) or (T, H, W, C).")
        frames_list = [frames[i] for i in range(frames.shape[0])]
    else:
        frames_list = list(frames)

    if len(frames_list) == 0:
        raise ValueError("`frames` is empty.")

    # Convert first frame to determine base size
    first_left_rgb = _to_hwc_rgb_uint8(frames_list[0])
    H, W = first_left_rgb.shape[:2]

    # Prepare goal (converted once, then resized as needed)
    goal_rgb = _to_hwc_rgb_uint8(goal_frame)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

    # Precompute a resized goal that matches (H, W) of the first frame
    goal_resized = cv2.resize(goal_rgb, (W, H), interpolation=cv2.INTER_AREA)

    for f in frames_list:
        left_rgb = _to_hwc_rgb_uint8(f)
        # If current frame size differs (rare, but robust), resize both to a common size
        if left_rgb.shape[:2] != (H, W):
            H, W = left_rgb.shape[:2]
            goal_resized = cv2.resize(goal_rgb, (W, H), interpolation=cv2.INTER_AREA)
            vw.release()
            vw = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))

        # OpenCV expects BGR
        left_bgr = left_rgb[:, :, ::-1]
        right_bgr = goal_resized[:, :, ::-1]

        side_by_side = np.concatenate([left_bgr, right_bgr], axis=1)
        vw.write(side_by_side)

    vw.release()
	
def load_offline_episode(data_path, episode_idx):
	"""Load a single episode from DinoWM dataset."""
	data_path = Path(data_path)

	# Load tensors
	states = torch.load(data_path / "states.pth").float()
	actions = torch.load(data_path / "rel_actions.pth").float()

	with open(data_path / "seq_lengths.pkl", "rb") as f:
		import pickle
		seq_lengths = pickle.load(f)

	# Get episode data
	seq_length = seq_lengths[episode_idx]
	episode_states = states[episode_idx, :seq_length]
	episode_actions = actions[episode_idx, :seq_length-1]

	# Load full video
	vid_path = data_path / "obses" / f"episode_{episode_idx:03d}.mp4"
	vr = VideoReader(str(vid_path), num_threads=1, ctx=cpu(0))

	# Get all frames
	all_frames = []
	for i in range(len(vr)):
		all_frames.append(vr[i].asnumpy())
	all_frames = np.array(all_frames)

	# Get initial state
	initial_state = episode_states[0].numpy()

	return all_frames, initial_state, seq_length


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Flexible objective that supports dict-based obs (with 'visual'/'proprio') or raw tensors.
        - If tensors are provided, computes MSE over all non-batch dims.
        - If dicts are provided, computes MSE on the last time step for any overlapping keys.

        Args:
            z_obs_pred: dict or tensor
            z_obs_tgt: dict or tensor
        Returns:
            loss: tensor (B, )
        """

        def to_dict(x):
            if isinstance(x, dict):
                return x
            return {"visual": x}

        pred_d = to_dict(z_obs_pred)
        tgt_d = to_dict(z_obs_tgt)

        def align_time_and_last(pred, tgt):
            """Ensure both have an explicit time dimension and take last from pred.
            Accepts shapes like (B, T, ...) or (B, ...). Returns tensors with matching time dims.
            """
            if pred.ndim == tgt.ndim + 1:
                # pred has time dim, tgt doesn't
                tgt = tgt.unsqueeze(1)
            elif tgt.ndim == pred.ndim + 1:
                # tgt has time dim, pred doesn't
                pred = pred.unsqueeze(1)
            elif pred.ndim == tgt.ndim:
                # neither or both have time; no change
                pass
            else:
                raise ValueError(f"Dim mismatch for loss: pred {tuple(pred.shape)} vs tgt {tuple(tgt.shape)}")

            # Take last step from pred if time exists
            if pred.ndim >= 3:  # (B, T, ...)
                pred_last = pred[:, -1:]
            else:
                pred_last = pred

            # Match target time dim to pred_last
            if tgt.ndim == pred_last.ndim and tgt.shape[1] != pred_last.shape[1]:
                tgt = tgt[:, -1:]

            return pred_last, tgt

        losses = []

        # Visual stream (if present)
        if "visual" in pred_d and "visual" in tgt_d and pred_d["visual"] is not None and tgt_d["visual"] is not None:
            v_pred, v_tgt = align_time_and_last(pred_d["visual"], tgt_d["visual"])
            losses.append(
                metric(v_pred, v_tgt).mean(dim=tuple(range(1, v_pred.ndim)))
            )

        # Proprio stream (optional)
        if "proprio" in pred_d and "proprio" in tgt_d and pred_d["proprio"] is not None and tgt_d["proprio"] is not None:
            p_pred, p_tgt = align_time_and_last(pred_d["proprio"], tgt_d["proprio"])
            losses.append(
                alpha * metric(p_pred, p_tgt).mean(dim=tuple(range(1, p_pred.ndim)))
            )

        if not losses:
            # Fallback: treat inputs as raw tensors if no overlapping keys
            if isinstance(z_obs_pred, torch.Tensor) and isinstance(z_obs_tgt, torch.Tensor):
                v_pred, v_tgt = align_time_and_last(z_obs_pred, z_obs_tgt)
                return metric(v_pred, v_tgt).mean(dim=tuple(range(1, v_pred.ndim)))
            raise ValueError("objective_fn_last: No overlapping keys to compute loss.")

        loss = sum(losses)
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError



def load_offline_episode_deformable(data_path, episode_idx):
	"""Load a single episode from DinoWM deformable dataset.

	This mirrors `load_offline_episode` but uses the deformable data layout:

	- Root contains `states.pth` and `actions.pth`.
	- Each episode lives under a folder like `000000/` with an `obses.pth` file.
	- Returns (all_frames, initial_state, seq_length) just like `load_offline_episode`.
	"""
	data_path = Path(data_path)
	# Load tensors (states required, actions used to derive a compact 4D initial state)
	states = torch.load(data_path / "states.pth").float()
	actions = torch.load(data_path / "actions.pth").float()

	# Episode slices
	episode_states = states[episode_idx]  # shapes like [T, P, 4] or [T, D]
	episode_actions = actions[episode_idx]  # expected [T, 4]

	# Identify valid (non-null) timesteps: finite and not-all-zero
	es = episode_states
	if es.ndim == 3:
		# [T, P, 4]
		finite_mask = torch.isfinite(es).all(dim=2).all(dim=1)  # [T]
		nonzero_mask = (es.abs().sum(dim=2).sum(dim=1) > 0)  # [T]
	elif es.ndim == 2:
		# [T, D]
		finite_mask = torch.isfinite(es).all(dim=1)
		nonzero_mask = (es.abs().sum(dim=1) > 0)
	else:
		# Fallback: consider all valid
		finite_mask = torch.ones((es.shape[0],), dtype=torch.bool)
		nonzero_mask = finite_mask
	valid_state_mask = (finite_mask & nonzero_mask).cpu().numpy()

	# Valid action timesteps
	ea = episode_actions
	if ea.ndim == 2:
		a_finite_mask = torch.isfinite(ea).all(dim=1)
		a_nonzero_mask = (ea.abs().sum(dim=1) > 0)
		valid_action_mask = (a_finite_mask & a_nonzero_mask).cpu().numpy()
	else:
		valid_action_mask = np.ones((es.shape[0],), dtype=bool)

	# Choose valid frame indices (prefer states validity)
	valid_idx = np.where(valid_state_mask)[0]
	if valid_idx.size == 0:
		# Fallback to any finite action indices
		valid_idx = np.where(valid_action_mask)[0]
	# If still none, default to first three timesteps if available
	if valid_idx.size == 0:
		valid_idx = np.arange(min(3, es.shape[0]))

	# Enforce the user's note: trajectories of nominal length 5 with nulls → keep first 3 valid
	if valid_idx.size > 3:
		valid_idx = valid_idx[:3]

	seq_length = int(valid_idx.size)

	# Derive a compact 4D initial state from the first valid action
	# Use [sx, sz, ex, ez] from actions (shape [4])
	first_valid_action_idx = int(np.where(valid_action_mask)[0][0]) if np.any(valid_action_mask) else int(valid_idx[0])
	initial_state = episode_actions[first_valid_action_idx].cpu().numpy().reshape(-1)  # (4,)

	# Load all frames from per-episode obses.pth (THWC), then select valid indices
	episode_dir = data_path / f"{episode_idx:06d}"
	obses_path = episode_dir / "obses.pth"
	if not obses_path.exists():
		raise FileNotFoundError(f"Could not find frames for episode {episode_idx} at {obses_path}")

	frames = torch.load(obses_path)
	if isinstance(frames, torch.Tensor):
		frames_np = frames.cpu().numpy()
	else:
		frames_np = np.array(frames)

	# Bound valid indices by available frames
	valid_idx = valid_idx[valid_idx < frames_np.shape[0]]
	all_frames = frames_np[valid_idx]
	seq_length = int(len(valid_idx))

	return all_frames, initial_state, seq_length



def main():
	# Configuration
	data_path = "/home/hrish/vjepa2_custom_tasks/data/deformable/rope"
	episode_idx = 0 
	goal_update_steps = 50  # Update goal every N planning steps
	goal_jump = 5  # Jump forward by M frames when updating goal
	output_video = "trajectory.mp4"
	debug_mode = False  # Enable debug visualizations

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	wm_path = "/home/hrish/vjepa2_custom_tasks/ckpts/outputs/vjep2/checkpoints/latest.pt"
	model_path = Path(wm_path).parent
	print(f"Looking for config in: {model_path}")
	config_path = model_path / "params-pretrain.yaml"
	# config_path = "/home/hrish/vjepa2_custom_tasks/configs/train/vitg16/dinowm-deformable-rope-256px-8f.yaml"
	with open(config_path, "r") as f:
		model_cfg = OmegaConf.load(f)

	# Extract config values like in train.py
	cfgs_model = model_cfg.get("model")
	use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
	model_name = cfgs_model.get("model_name")
	pred_depth = cfgs_model.get("pred_depth")
	pred_num_heads = cfgs_model.get("pred_num_heads", None)
	pred_embed_dim = cfgs_model.get("pred_embed_dim")
	pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
	uniform_power = cfgs_model.get("uniform_power", False)
	use_rope = cfgs_model.get("use_rope", False)
	use_silu = cfgs_model.get("use_silu", False)
	use_pred_silu = cfgs_model.get("use_pred_silu", False)
	wide_silu = cfgs_model.get("wide_silu", True)
	use_extrinsics = cfgs_model.get("use_extrinsics", False)
	action_embed_dim = cfgs_model.get("action_embed_dim", 4) # 7 for droid tasks, 2 for pushT, 4 for deformable tasks
	state_embed_dim = cfgs_model.get("state_embed_dim", 7860)

	# Get data config values
	cfgs_data = model_cfg.get("data")
	tubelet_size = cfgs_data.get("tubelet_size")
	crop_size = cfgs_data.get("crop_size", 256)
	patch_size = cfgs_data.get("patch_size")
	dataset_fpcs = cfgs_data.get("dataset_fpcs")
	max_num_frames = max(dataset_fpcs)

	# Get meta config values
	cfgs_meta = model_cfg.get("meta")
	use_sdpa = cfgs_meta.get("use_sdpa", False)

	# Get loss config values
	cfgs_loss = model_cfg.get("loss")
	normalize_reps = cfgs_loss.get("normalize_reps")
	auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
	tokens_per_frame = int((crop_size // patch_size) ** 2)

	# data aug
	cfgs_data_aug = model_cfg.get("data_aug")
	horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
	ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
	rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
	motion_shift = cfgs_data_aug.get("motion_shift", False)
	reprob = cfgs_data_aug.get("reprob", 0.0)
	use_aa = cfgs_data_aug.get("auto_augment", False)

	transforms = make_transforms(
		random_horizontal_flip=horizontal_flip,
		random_resize_aspect_ratio=ar_range,
		random_resize_scale=rr_scale,
		reprob=reprob,
		auto_augment=use_aa,
		motion_shift=motion_shift,
		crop_size=crop_size,
	)

	# Initialize model with parameters from config
	encoder, predictor = init_video_model(
		uniform_power=uniform_power,
		device=device,
		patch_size=patch_size,
		# Use the configured max number of frames to avoid OOM
		max_num_frames=max_num_frames,
		tubelet_size=tubelet_size,
		model_name=model_name,
		crop_size=crop_size,
		pred_depth=pred_depth,
		pred_num_heads=pred_num_heads,
		pred_embed_dim=pred_embed_dim,
		action_embed_dim=action_embed_dim, # 7 for droid tasks, 2 for pushT, 4 for deformable tasks
		state_embed_dim=state_embed_dim,
		pred_is_frame_causal=pred_is_frame_causal,
		use_extrinsics=use_extrinsics,
		use_sdpa=use_sdpa,
		use_silu=use_silu,
		use_pred_silu=use_pred_silu,
		wide_silu=wide_silu,
		use_rope=use_rope,
		use_activation_checkpointing=use_activation_checkpointing,
	)

	target_encoder = copy.deepcopy(encoder)

	# Load checkpoint 
	(
		encoder,
		predictor,
		target_encoder,
		optimizer,
		scaler,
		start_epoch,
	) = load_checkpoint(
		r_path=wm_path,
		encoder=encoder,
		predictor=predictor,
		target_encoder=target_encoder,
		opt=None,
		scaler=None,
	)

	encoder.eval()
	predictor.eval()
	encoder.to(device)
	predictor.to(device)

	# Create world model
	tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)

	world_model = WorldModel(
		encoder=encoder,
		predictor=predictor,
		tokens_per_frame=tokens_per_frame,
		transform=transforms,
		mpc_args={
			"rollout": 2,
			"samples": 3,
			"topk": 3,
			"cem_steps": 1,
			"momentum_mean": 0.15,
			"momentum_mean_gripper": 0.15,
			"momentum_std": 0.75,
			"momentum_std_gripper": 0.15,
			"maxnorm": 0.075,
			"verbose": False
		},
		normalize_reps=True,
		device=device
	)



	data_loader, dist_sampler, dataset = init_data(
		data_path=data_path,
		batch_size=1,
		frames_per_clip=max_num_frames,
		fps=model_cfg.data.fps,
		crop_size=model_cfg.data.crop_size,
		rank=0,
		world_size=1,
		camera_views=1,
		stereo_view=False,
		drop_last=True,
		num_workers=10,
		pin_mem=True,
		persistent_workers=True,
		collator=None,
		transform=None,
		camera_frame=False,
		tubelet_size=2,
		frameskip=model_cfg.data.frameskip,  # Separate parameter for data subsampling
            
	)
      
	# print(next(iter(data_loader)))
      
	  
	train_dset, val_dset = split_traj_datasets(
		dataset, train_fraction=0.9, random_seed=42
	)
      
	# general_seed = 7
	# eval_seed = 42
      
	# # seed(general_seed)
      
	observations, states, actions, env_info = sample_traj_segment_from_dset(val_dset)




	# one_ep_states = states[0]  # (T, D) or (T, P, 4)
	# one_ep_actions = actions[0]  # (T, 4)


	initial_state = states[0][0] # (D,) or (4,)
	all_frames = observations[0].numpy()  # (T, H, W, C)
      
	print(all_frames[2].mean())


	print("all_frames", all_frames.shape, "initial_state", initial_state.shape)


	config = {
		"env": {
			"name": "deformable_env",
			"args": [],
			"kwargs": {
				"object_name": "rope"
			},
			"load_dir": "",
			"dataset": {
				"_target_": "datasets.deformable_env_dset.load_deformable_dset_slice_train_val",
				"n_rollout": None,
				"normalize_action": "${normalize_action}",
				"data_path": "${oc.env:DATASET_DIR}/deformable",
				"object_name": "rope",
				"split_ratio": 0.9,
				"transform": {
					"_target_": "datasets.img_transforms.default_transform",
					"img_size": "${img_size}"
				}
			},
			"decoder_path": None,
			"num_workers": 0
		}
	}
		

	# Create environment
	# env = gym.make(env_config["name"], max_episode_steps=1000, **env_config["kwargs"])
	env = gym.make(
		config["env"]["name"],
		*config["env"]["args"],
		**config["env"]["kwargs"]
	)
	

	env.prepare(1, initial_state)
	# env.unwrapped._set_state(initial_state)

	# Initialize goal tracking
	current_goal_idx = goal_jump  # Start with first goal

	# Set initial goal
	initial_frame = all_frames[0]


	current_goal_frame = all_frames[min(current_goal_idx, len(all_frames) - 1)]

	current_goal_state = initial_state  # Will be updated

	# Prepare observations
	current_obs = {"visual": initial_frame, "proprio": initial_state}
	goal_obs = {"visual": current_goal_frame, "proprio": current_goal_state, "mode": 'constant'}

	# Plan and execute across MPC iterations
	all_exec_frames = [initial_frame]  # accumulate executed frames
	all_exec_actions = []              # accumulate executed actions
	prev_memo = None                   # warm-start tail for next iteration

	max_steps = 5  # mpc iterations
	cem_steps = 5
	cem_config = {
		"horizon": 1,  # plan horizon; execute 1 step each MPC
		"samples": 20,
		"topk": 10,
		"cem_steps": cem_steps,
		"var_scale": 1,
	}

	for mpc_iter in range(max_steps):  # max mpc iterations
		print(f"MPC step {mpc_iter+1}/{max_steps}, current goal idx: {current_goal_idx}/{len(all_frames)-1}")

		current_goal_frame = all_frames[current_goal_idx]

		# Update goal observation for model
		goal_obs = {"visual": current_goal_frame, "proprio": current_goal_state, "mode": 'constant'}

		# Run one MPC iteration; take 1 action; keep tail as warm start
		taken_actions, memo_tail, new_frames, logs = world_model.perform_mpc_iter(
			env=env,
			current_obs=current_obs,
			goal_obs=goal_obs,
			objective_fn_latent=create_objective_fn(1.0, 2.0, mode="last"),
			cem_config=cem_config,
			n_taken_actions=1,
			memo_actions=prev_memo,
		)

		# Accumulate actions and frames
		all_exec_actions.extend([a.detach().cpu().numpy() for a in taken_actions])
		all_exec_frames.extend(list(new_frames))

		print(f"Taken action(s) this iter: {taken_actions.shape[0]}, total so far: {len(all_exec_actions)}")

		# Update current observation for next iteration
		current_obs = {"visual": new_frames[-1], "proprio": current_goal_state}

		# Carry over CEM warm-start tail
		prev_memo = memo_tail

		# Optionally update the goal index periodically
		if (mpc_iter + 1) % goal_update_steps == 0:
			current_goal_idx = min(current_goal_idx + goal_jump, len(all_frames) - 1)

		# Save one final side-by-side video for the whole executed trajectory
		save_side_by_side_video(all_exec_frames, current_goal_frame, output_video, 10, "mp4v")
		print(f"Saved side-by-side trajectory video to {output_video}")

		# Final evaluation: rollout executed actions in the env and print chamfer metrics
		try:
			exec_actions_np = np.stack(all_exec_actions, axis=0)
			#print things going in
			print("Executing actions in env for final evaluation:", exec_actions_np.shape, exec_actions_np)
			_, e_states = env.rollout(1, initial_state, exec_actions_np)

			# Get the last executed state from the rollout: expected shape (N, 4)
			last_state_np = e_states[-1]
			if isinstance(last_state_np, torch.Tensor):
				last_state_np = last_state_np.detach().cpu().numpy()
			# If flattened, reshape to (N, 4)
			if last_state_np.ndim == 1 and last_state_np.size % 4 == 0:
				last_state_np = last_state_np.reshape(-1, 4)
			elif last_state_np.ndim == 2 and last_state_np.shape[-1] != 4 and (last_state_np.size % 4 == 0):
				last_state_np = last_state_np.reshape(-1, 4)

			# Derive goal state from dataset at the current goal index
			# Dataset states are flattened (P*4); reshape to (P, 4)
			goal_idx = int(min(current_goal_idx, states.shape[1] - 1))
			goal_state_flat = states[0, goal_idx]
			if isinstance(goal_state_flat, torch.Tensor):
				goal_state_flat = goal_state_flat.detach().cpu().numpy()
			goal_state_np = goal_state_flat.reshape(-1, 4)

			# Convert to torch tensors and batch: [1, N, D]
			g = torch.as_tensor(goal_state_np, dtype=torch.float32)
			c = torch.as_tensor(last_state_np, dtype=torch.float32)
			if g.ndim == 2:
				g = g.unsqueeze(0)
			if c.ndim == 2:
				c = c.unsqueeze(0)

			# Compute Chamfer distance (uses only first 3 dims internally)
			cd = chamfer_distance(g, c)
			print("Final states shape:", e_states.shape)
			print("Goal state shape:", goal_state_np.shape, "Last state shape:", last_state_np.shape)
			print("Chamfer distance (goal vs executed last state):", float(cd.item()))

			# Also use env.eval_state for consistency and to verify shapes
			metrics = env.eval_state(goal_state_np, last_state_np)
			print("Eval metrics:", metrics)
		except Exception as ex:
			print("Final evaluation failed:", ex)


if __name__ == "__main__":
	main()
