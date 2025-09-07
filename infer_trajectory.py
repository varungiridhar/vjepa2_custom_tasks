import os
import sys
import cv2
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from decord import VideoReader, cpu
from omegaconf import OmegaConf
import copy

sys.path.insert(0, "..")

from app.vjepa_droid.utils import init_video_model, load_pretrained, load_checkpoint
from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.world_model_wrapper import WorldModel
import env  # This registers the pusht environment with gymnasium


def world_to_image_coords(world_pos, image_size=224, world_size=512):
	"""Convert world coordinates to image coordinates."""
	x, y = world_pos
	# Scale from world coordinates (0-512) to image coordinates (0-image_size)
	scale = image_size / world_size
	return (int(x * scale), int(y * scale))


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


def main():
	# Configuration
	data_path = "/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/dinowm_data/datasets/pusht_noise/train"
	episode_idx = 0
	goal_update_steps = 50  # Update goal every N planning steps
	goal_jump = 2  # Jump forward by M frames when updating goal
	output_video = "trajectory.mp4"
	debug_mode = True  # Enable debug visualizations

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	wm_path = "/storage/home/hcoda1/6/vgiridhar6/scratch/vjepa2/082725/latest.pt"
	model_path = Path(wm_path).parent
	print(f"Looking for config in: {model_path}")
	config_path = model_path / "params-pretrain.yaml"
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
		max_num_frames=512,
		tubelet_size=tubelet_size,
		model_name=model_name,
		crop_size=crop_size,
		pred_depth=pred_depth,
		pred_num_heads=pred_num_heads,
		pred_embed_dim=pred_embed_dim,
		action_embed_dim=7,
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

	# Create transform
	crop_size = 256
	transform = make_transforms(
		random_horizontal_flip=False,
		random_resize_aspect_ratio=(1., 1.),
		random_resize_scale=(1., 1.),
		reprob=0.,
		auto_augment=False,
		motion_shift=False,
		crop_size=crop_size,
	)

	# Create world model
	tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
	world_model = WorldModel(
		encoder=encoder,
		predictor=predictor,
		tokens_per_frame=tokens_per_frame,
		transform=transform,
		mpc_args={
			"rollout": 5,
			"samples": 30,
			"topk": 3,
			"cem_steps": 5,
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

	# Load offline episode
	all_frames, initial_state, seq_length = load_offline_episode(
		data_path, episode_idx
	)

	# Create environment
	env_config = {
		"name": "pusht",
		"kwargs": {
			"with_velocity": True,
			"with_target": True
		}
	}

	# Create environment
	env = gym.make(env_config["name"], max_episode_steps=1000, **env_config["kwargs"])
	# Set initial state
	padded_state = np.pad(initial_state, (0, 2), mode='constant')
	env.reset()
	env.unwrapped._set_state(padded_state)

	# Initialize goal tracking
	current_goal_idx = goal_jump  # Start with first goal
	goal_history = []  # Track (frame_idx, goal_state) for debugging

	# Set initial goal
	initial_frame = all_frames[0]
	current_goal_frame = all_frames[min(current_goal_idx, len(all_frames) - 1)]
	current_goal_state = initial_state  # Will be updated

	# Prepare observations
	current_obs = {"visual": initial_frame, "proprio": padded_state}
	goal_obs = {"visual": current_goal_frame, "proprio": np.pad(current_goal_state, (0, 2), mode='constant')}

	# Plan and execute
	frames = [initial_frame]
	max_steps = 100

	# Repeat the same action for N env steps before replanning
	action_repeat = 5
	last_action = None  # cached action to repeat

	for step in range(max_steps):
		# Update goal every N steps
		if step > 0 and step % goal_update_steps == 0:
			current_goal_idx = min(current_goal_idx + goal_jump, len(all_frames) - 1)
			current_goal_frame = all_frames[current_goal_idx]

			# Record goal update for debugging
			goal_history.append({
				'step': step,
				'goal_frame_idx': current_goal_idx,
				'goal_state': current_goal_state.tolist()  # Convert numpy array to list
			})

			print(f"Step {step}: Updated goal to frame {current_goal_idx}")

		# Update goal observation for model
		goal_obs = {"visual": current_goal_frame, "proprio": np.pad(current_goal_state, (0, 2), mode='constant')}

		# Encode current and goal
		with torch.no_grad():
			current_rep = world_model.encode(current_obs["visual"])
			goal_rep = world_model.encode(goal_obs["visual"])

		# Plan once every `action_repeat` steps; reuse the same action otherwise
		if (step % action_repeat == 0) or (last_action is None):
			action = world_model.infer_next_action(
				current_rep,
				torch.tensor(padded_state).float().to(device).unsqueeze(0),
				goal_rep
			)
			# Detach for safety when reusing across steps
			last_action = action.detach()
		else:
			action = last_action
		
		# Debug: Predict next pose using the chosen action
		if debug_mode:
			action_scale = 1000.0
			print(f"Step {step}: Current pose: {padded_state[:2]}")
			print(f"Step {step}: MPC action: {action[0].cpu().numpy()}")
			print(f"Step {step}: Scaled action: {action_scale * action[0].cpu().numpy()}")
			
			predicted_next_pose = world_model.predict_next_pose(
				current_rep.unsqueeze(0),
				torch.tensor(padded_state).unsqueeze(0).unsqueeze(0).float().to(device),
				action_scale * action[0].unsqueeze(0).unsqueeze(0)  # Use the first (and only) action
			)
			
			print(f"Step {step}: Predicted pose: {predicted_next_pose.cpu().numpy()}")
			print(f"Step {step}: Pose difference: {predicted_next_pose.cpu().numpy() - padded_state}")
			
			debug_info = {
				'mpc_action': action[0].cpu().numpy(),
				'predicted_pose': predicted_next_pose.cpu().numpy(),
				'current_pose': padded_state.copy()
			}
		else:
			debug_info = None

		# Execute action (use first 2 dimensions)
		xy_action = action[0, :2].cpu().numpy()
		obs, reward, terminated, truncated, info = env.step(xy_action)

		# Update state
		padded_state = info["state"]
		current_obs = {"visual": obs["visual"], "proprio": padded_state}

		# Apply debug visualizations if enabled
		if debug_mode and debug_info is not None:
			debug_frame = obs["visual"].copy()
			
			# Draw MPC action arrow (red)
			debug_frame = draw_mpc_action_arrow(
				debug_frame, 
				debug_info['current_pose'][:2], 
				debug_info['mpc_action'],
				color=(0, 0, 255),  # Red arrow for MPC action
				arrow_length=40
			)
			
			# Draw predicted next pose (orange circle) - HARDCODED TO (0,0) FOR TESTING
			debug_frame = draw_predicted_pose(
				debug_frame,
				debug_info['predicted_pose'][:2],  # Use predicted pose from model
				color=(212, 129, 11),  # Orange circle for predicted pose
				radius=6
			)
			
			# Save debug frame instead of original
			frames.append(debug_frame)
		else:
			# Save original frame
			frames.append(obs["visual"])

		if terminated or truncated:
			break

	# Create video with side-by-side view (trajectory | goal image)
	height, width = frames[0].shape[:2]

	# Create side-by-side video (trajectory on left, goal on right)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(output_video, fourcc, 10, (width * 2, height))  # Twice as wide

	# Track current goal frame index for video
	current_video_goal_idx = goal_jump

	for i, frame in enumerate(frames):
		# Update goal frame in video every goal_update_steps
		if i > 0 and i % goal_update_steps == 0:
			current_video_goal_idx = min(current_video_goal_idx + goal_jump, len(all_frames) - 1)

		# Get current goal frame for this step
		current_goal_frame = all_frames[min(current_video_goal_idx, len(all_frames) - 1)]
		goal_img_bgr = cv2.cvtColor(current_goal_frame, cv2.COLOR_RGB2BGR)

		# Convert trajectory frame to BGR
		traj_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		
		# Create side-by-side frame
		combined_frame = np.hstack([traj_bgr, goal_img_bgr])

		video.write(combined_frame)

	video.release()

	print(f"Saved side-by-side trajectory video to {output_video}")
	print(f"Goal updates occurred at steps: {[g['step'] for g in goal_history]}")


if __name__ == "__main__":
	main()
