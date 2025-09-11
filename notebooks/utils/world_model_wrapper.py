from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch.nn.functional as F
import torch
from einops import rearrange


class WorldModel(object):

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def encode(self, image):
        clip = np.expand_dims(image, axis=0)
        clip = self.transform(clip)[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    # def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):

    #     def step_predictor(reps, actions, poses):
    #         B, T, N_T, D = reps.size()
    #         reps = reps.flatten(1, 2)
    #         next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
    #         if self.normalize_reps:
    #             next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
    #         next_rep = next_rep.view(B, 1, N_T, D)
    #         next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
    #         return next_rep, next_pose

    #     mpc_action = cem(
    #         context_frame=rep,
    #         context_pose=pose,
    #         goal_frame=goal_rep,
    #         world_model=step_predictor,
    #         close_gripper=close_gripper,
    #         **self.mpc_args,
    #     )[0]

    #     return mpc_action
    
    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Batched latent rollout to match CEM usage.
        Inputs:
        - z0:      (B, Ztokens, D) initial latent tokens (current frame(s)).
                   If multiple frames are present, the last frame is used.
        - actions: (B, H, A) horizon action sequences.
        Returns:
        - z_T:     (B, Ztokens, D) final predicted latent tokens (same length as z0).
        """
        # Validate inputs
        assert isinstance(z0, torch.Tensor) and isinstance(actions, torch.Tensor), "z0 and actions must be tensors"
        device = z0.device
        B, Z, D = z0.shape
        H = actions.shape[1]
        P = int(self.tokens_per_frame)

        if Z % P != 0:
            raise ValueError(f"rollout: z0 tokens ({Z}) must be divisible by tokens_per_frame ({P})")

        # Use only the last frame worth of tokens as context
        x = z0[:, -P:, :]  # (B, P, D)

        # Infer predictor IO for AC models
        pred_state_in = getattr(getattr(self.predictor, "state_encoder", None), "in_features", None)
        use_extr = bool(getattr(self.predictor, "use_extrinsics", False))
        pred_extr_in = getattr(getattr(self.predictor, "extrinsics_encoder", None), "in_features", None) if use_extr else None

        # Zero states/extrinsics for planning
        states = torch.zeros(B, H, pred_state_in or 1, device=device, dtype=actions.dtype) if pred_state_in is not None else None
        extr = torch.zeros(B, H, pred_extr_in, device=device, dtype=actions.dtype) if use_extr else None

        # Step the predictor auto-regressively for H steps
        for t in range(H):
            a_t = actions[:, t : t + 1, :]                    # (B, 1, A)
            s_t = states[:, t : t + 1, :] if states is not None else None
            e_t = extr[:, t : t + 1, :] if extr is not None else None
            #print all shapes
            # print(f"Rollout step {t}: x={x.shape}, a_t={a_t.shape}, s_t={s_t.shape if s_t is not None else None}, e_t={e_t.shape if e_t is not None else None}")
            x = self.predictor(x, a_t, s_t, e_t)              # (B, P, D)
            if self.normalize_reps:
                x = F.layer_norm(x, (x.size(-1),))

        # Match output token length expected by callers
        if Z == P:
            return x
        else:
            # If z0 contained multiple frames, repeat the final frame tokens to match shape
            repeats = Z // P
            return x.repeat(1, repeats, 1)

    @torch.no_grad()
    def perform_mpc_iter(
        self,
        env,
        current_obs: Dict[str, Any],     # expects current_obs["visual"]
        goal_obs: Dict[str, Any],        # expects goal_obs["visual"]
        objective_fn_latent,  # (pred, goal) -> loss
        cem_config: Dict[str, Any],
        n_taken_actions: int = 1,
        memo_actions: Optional[torch.Tensor] = None,  # (1, t0, A) or None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Dict[str, float]]:
        """
        One MPC iteration:
        1) Plan with CEM in latent space
        2) Execute first n_taken_actions on the env
        3) Evaluate latent loss vs goal
        4) Return taken actions, memo tail, frames, and logs

        Returns:
        taken_actions: (K, A) torch.float32 on CUDA (K = n_taken_actions)
        memo_tail:     (1, H-K, A) torch.float32 on CUDA, or None if empty
        frames:        np.ndarray (K, H, W, 3) RGB from env.step_multiple
        logs:          dict of scalars {"latent_loss": float, "executed_steps": float}
        """

        device = torch.device(cem_config.get("device", self.device))

        # --- Encode current & goal ---
        cur_rep  = self.encode(current_obs["visual"]).to(device)
        goal_rep = self.encode(goal_obs["visual"]).to(device)


        # --- Plan with CEM ---
        planned_mean, _ = self.cem(
            context_latent=cur_rep,
            goal_latent=goal_rep,
            objective_fn_latent=objective_fn_latent,
            action_dim=4,  # REQUIRED
            horizon=cem_config.get("horizon", 8),
            cem_steps=cem_config.get("cem_steps", 10),
            samples=cem_config.get("samples", 400),
            topk=cem_config.get("topk", 10),
            var_scale=cem_config.get("var_scale", 1.0),
            init_actions=memo_actions,
        )

        # --- Split actions ---
        K = int(n_taken_actions)
        taken_actions = planned_mean[:K]
        memo_tail     = planned_mean[K:].unsqueeze(0) if planned_mean.size(0) > K else None

        # --- Execute in env ---
        actions_list = [a.detach().cpu().numpy() for a in taken_actions]
        obses, _, _, _ = env.step_multiple(actions_list)
        frames = obses["visual"]  # np.ndarray (K, H, W, 3) RGB

        # --- Evaluate latent loss on last frame ---
        last_rep = self.encode(frames[-1]).to(device)  # (1, Z, D)
        # Wrap latents as dicts with a singleton time dimension for compatibility
        latent_loss_vec = objective_fn_latent(
            {"visual": last_rep.unsqueeze(1)},
            {"visual": goal_rep.unsqueeze(1)}
        )
        latent_loss = latent_loss_vec.mean().item()

        # --- Logs ---
        logs = {
            "latent_loss": float(latent_loss),
            "executed_steps": float(K),
        }

        return taken_actions, memo_tail, frames, logs
    

    @torch.no_grad()
    def cem(
            self,
            context_latent: torch.Tensor,
            goal_latent: torch.Tensor,
            objective_fn_latent,
            action_dim: int,
            horizon: int = 8,
            cem_steps: int = 10,
            samples: int = 400,
            topk: int = 10,
            var_scale: float = 1.0,
            init_actions: torch.Tensor | None = None):
        """
        context_latent: (1, Ztokens, D)
        goal_latent:    (1, Ztokens, D)
        init_actions:   (1, t0, A) optional warm start prefix (t0 <= horizon)
        Returns:
        best_mean: (H, A) mean sequence (use first action for MPC)
        best_sigma:(H, A) std sequence
        """

        device = context_latent.device
        S = samples
        H = horizon
        A = action_dim

        # --- initialize mean/std ---
        if init_actions is not None and init_actions.shape[1] > 0:
            t0 = min(init_actions.shape[1], H)
            mean = torch.zeros(H, A, device=device)
            mean[:t0] = init_actions[0, :t0]
        else:
            mean = torch.zeros(H, A, device=device)

        sigma = torch.ones(H, A, device=device) * var_scale

        # Cache repeated latents for batch eval
        z0 = context_latent.expand(S, -1, -1)  # (S, Ztokens, D)
        g  = goal_latent                       # (1, Ztokens, D)

        for it in range(cem_steps):
            # --- sample action sequences ~ N(mean, sigma) ---
            eps = torch.randn(S, H, A, device=device)
            action_seqs = eps * sigma[None, ...] + mean[None, ...]  # (S,H,A)

            # Optional: clamp or normalize actions
            if var_scale is not None:
                # Per-step clip to L2 radius
                norms = action_seqs.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (S,H,1)
                scale = torch.minimum(torch.ones_like(norms), var_scale / norms)
                action_seqs = action_seqs * scale

            # --- rollout in latent space (batched) ---
            final_latents = self.rollout(z0, action_seqs)   # (S, Ztokens, D)

            # --- score vs goal ---
            # Wrap as dicts with a singleton time dimension for objective compatibility
            losses = objective_fn_latent(
                {"visual": final_latents.unsqueeze(1)},
                {"visual": g.unsqueeze(1)}
            )  # (S,)
            topk_idx = torch.topk(-losses, k=topk).indices  # largest (-loss) == smallest loss

            elite_actions = action_seqs[topk_idx]           # (topk, H, A)

            # --- refit Gaussian ---
            mean = elite_actions.mean(dim=0)          # (H, A)
            sigma  = elite_actions.std(dim=0).clamp_min(1e-6)

        # final pick: return mean (and sigma)
        return mean, sigma
