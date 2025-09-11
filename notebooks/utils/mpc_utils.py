from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from einops import repeat


from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor



@torch.no_grad()
def cem(context_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        world_model,
        objective_fn_latent,
        action_dim: int,
        horizon: int = 8,
        cem_steps: int = 10,
        samples: int = 400,
        topk: int = 10,
        var_scale: float = 1.0,
        momentum_mean: float = 0.15,
        momentum_std: float = 0.15,
        maxnorm: float = 0.05,
        verbose: bool = False,
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

    prev_mean = mean.clone()
    prev_sigma = sigma.clone()

    for it in range(cem_steps):
        # --- sample action sequences ~ N(mean, sigma) ---
        eps = torch.randn(S, H, A, device=device)
        action_seqs = eps * sigma[None, ...] + mean[None, ...]  # (S,H,A)

        # Optional: clamp or normalize actions
        if maxnorm is not None:
            # Per-step clip to L2 radius
            norms = action_seqs.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (S,H,1)
            scale = torch.minimum(torch.ones_like(norms), maxnorm / norms)
            action_seqs = action_seqs * scale

        # --- rollout in latent space (batched) ---
        final_latents = world_model.rollout_latent(z0, action_seqs)   # (S, Ztokens, D)

        # --- score vs goal ---
        losses = objective_fn_latent(final_latents, g, reduction="mean")  # (S,)
        topk_idx = torch.topk(-losses, k=topk).indices  # largest (-loss) == smallest loss

        elite_actions = action_seqs[topk_idx]           # (topk, H, A)

        # --- refit Gaussian ---
        elite_mean = elite_actions.mean(dim=0)          # (H, A)
        elite_std  = elite_actions.std(dim=0).clamp_min(1e-6)

        # Momentum smoothing (helps stability)
        mean  = (1 - momentum_mean) * elite_mean + momentum_mean * prev_mean
        sigma = (1 - momentum_std)  * elite_std  + momentum_std  * prev_sigma

        prev_mean = mean.clone()
        prev_sigma = sigma.clone()

        if verbose and (it % 2 == 0 or it == cem_steps - 1):
            print(f"[CEM] iter {it+1}/{cem_steps} | loss: {losses.mean().item():.4f} | "
                  f"elite: {losses[topk_idx].mean().item():.4f}")

    # final pick: return mean (and sigma)
    return mean, sigma