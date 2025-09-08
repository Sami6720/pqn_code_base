# analysis_helpers.py
# Helper functions for analyzing networks (effective ranks, NTK rank, dormancy, norms, Q stats)
# Works with JAX/Flax; optionally integrate with your Flax modules via `sow('intermediates', ...)`.

from typing import Callable, Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax


# --------------------------------------------------------------------------------------
# Generic utilities
# --------------------------------------------------------------------------------------

def l2(x: jnp.ndarray) -> jnp.ndarray:
    """L2 norm of an array (flattened)."""
    return jnp.linalg.norm(x.reshape(-1))


def _flatten_params(params: Any) -> Dict[str, jnp.ndarray]:
    """Flatten a PyTree of parameters to a dict with '/'-joined keys."""
    return flax.traverse_util.flatten_dict(params, sep='/')


def _unflatten_params(flat: Dict[str, jnp.ndarray]) -> Any:
    """Unflatten a dict of '/'-joined keys back into a PyTree."""
    return flax.traverse_util.unflatten_dict(flat, sep='/')


def ema_update(old: jnp.ndarray, new: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Exponential moving average: alpha * new + (1 - alpha) * old."""
    return alpha * new + (1.0 - alpha) * old


# --------------------------------------------------------------------------------------
# Effective rank (features or Gram matrices)
# --------------------------------------------------------------------------------------

def effective_rank(X: jnp.ndarray, delta: float = 0.01) -> jnp.ndarray:
    # X is [B, D] (rows = samples, cols = features); if more dims, flatten features
    assert X.ndim == 2

    s = jnp.linalg.svd(X, compute_uv=False, full_matrices=False)  # sorted desc
    s_sum = jnp.sum(s)

    # If all singular values are ~0, define rank as 0.
    def zero_rank(_):
        return jnp.array(0, dtype=jnp.int32)

    def nonzero_rank(_):
        csum = jnp.cumsum(s)/s_sum
        return jnp.argmax(csum >= (1.0 - delta)).astype(jnp.int32) + 1

    return jax.lax.cond(s_sum <= 0.0, zero_rank, nonzero_rank, operand=None)

def effective_rank_per_expert(Y: jnp.ndarray, delta: float = 0.01) -> jnp.ndarray:
    """
    Effective rank per expert for features shaped [B, N, D*] or [B, N, M, D].
    Flattens the last dims to [B, N, Dflat] and computes srank per expert.
    Returns [N] sranks (float32).
    """
    assert Y.ndim >= 3, "Y must be [B,N,*]"
    B, N = Y.shape[0], Y.shape[1]
    Y_flat = Y.reshape(B, N, -1)           # [B, N, Dflat]
    # compute per-expert effective rank

    # exp_feats: [B, Dflat]
    def srank_one(exp_feats: jnp.ndarray) -> jnp.ndarray:
        return effective_rank(exp_feats, delta)
    return jax.vmap(srank_one, in_axes=1, out_axes=0)(Y_flat)  # [N]


# --------------------------------------------------------------------------------------
# Empirical NTK effective rank
# --------------------------------------------------------------------------------------

def _flatten_out(q: jnp.ndarray) -> jnp.ndarray:
    """Flatten model output (e.g., Q-values) to a vector."""
    return q.reshape(-1)


def _mask_tree_like_params(param_tree: Any,
                           tree: Any,
                           mask_fn: Callable[[str], bool]) -> Any:
    """
    Zero-out leaves in `tree` according to a predicate on the flattened keys of `param_tree`.
    Useful to restrict Jacobians to a subset of parameters.
    """
    flat_p = _flatten_params(param_tree)
    flat_t = flax.traverse_util.flatten_dict(tree, sep='/')
    for k in flat_t.keys():
        if not mask_fn(k):
            flat_t[k] = jnp.zeros_like(flat_t[k])
    return _unflatten_params(flat_t)


def ntk_srank(apply_fn: Callable,
              variables: Dict[str, Any],
              batch_x: jnp.ndarray,
              delta: float = 0.01,
              per_action: bool = False,
              param_mask_fn: Optional[Callable[[str], bool]] = None) -> jnp.ndarray:
    """
    Empirical NTK effective rank on a minibatch.
    K_ij = <∂f(x_i)/∂θ, ∂f(x_j)/∂θ>, with f returning a 1D vector per sample.
    We add a batch dimension for apply_fn because the model expects batched inputs.
    """

    def f(params, x_single):
        # Ensure batched input: [1, ...]
        x_b = jnp.expand_dims(x_single, 0)
        q = apply_fn({"params": params, "batch_stats": variables.get("batch_stats")},
                     x_b, train=False)
        # Remove batch dim: now [A] (or [N, A] if per-expert), depending on the model
        q = q[0]
        return q if per_action else _flatten_out(q)

    jf = jax.jacrev(lambda p, x: f(p, x))  # ∂f/∂θ

    def jac_one(x_single):
        Jtree = jf(variables["params"], x_single)
        if param_mask_fn is not None:
            Jtree = _mask_tree_like_params(variables["params"], Jtree, param_mask_fn)
        flat, _ = ravel_pytree(Jtree)  # [P]
        return flat

    # Vmap over the (already) batched batch_x: each element is a single sample
    J = jax.vmap(jac_one)(batch_x)  # [B, P]
    K = J @ J.T                     # [B, B]
    return effective_rank(K, delta=delta)



# --------------------------------------------------------------------------------------
# Dormant neurons (per-layer)
# --------------------------------------------------------------------------------------

def dormant_fraction(acts: jnp.ndarray,
                     reduce_spatial: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fraction of τ-dormant units in a layer: units with mean |activation| below tau.
    Returns (fraction_dormant, per_unit_mean), where per_unit_mean is over batch & spatial.

    acts:
      Dense: [B, D]
      Conv/tokens: [B, H, W, C] or [B, M, D]
    """
    a = jnp.abs(acts)
    # if acts.ndim >= 3:  # conv or tokens
    #     if reduce_spatial:
    #         # average over all non-batch, non-channel dims except the last feature dim
    #         a = jnp.mean(a, axis=tuple(range(1, acts.ndim - 1)))  # -> [B, C/D]
    #     else:
    #         # average over spatial/tokens, keep feature dim
    #         a = a.reshape(a.shape[0], -1, a.shape[-1])  # [B, S, F]
    #         a = jnp.mean(a, axis=1)                     # [B, F]
    # elif acts.ndim == 2:  # [B, D]
    #     pass
    # else:
    #     a = a.reshape(a.shape[0], -1)

    per_unit = jnp.mean(a, axis=0)  # [units]
    return per_unit


def dormant_fraction_dict(intermediates: Dict[str, jnp.ndarray],
                          tau: float = 0.05,
                          reduce_spatial: bool = True) -> Dict[str, jnp.ndarray]:
    """
    Apply dormant_fraction to a dict of named activations (e.g., from `sow` intermediates).
    Returns {name: dormant_fraction}.
    """
    out = {}
    for k, v in intermediates.items():
        try:
            frac, _ = dormant_fraction(
                v, tau=tau, reduce_spatial=reduce_spatial)
            out[k] = frac
        except Exception:
            # Skip non-array items or unsupported shapes
            continue
    return out


# --------------------------------------------------------------------------------------
# MoE routing & phi norms
# --------------------------------------------------------------------------------------

def routing_utilization(dispatch: jnp.ndarray) -> jnp.ndarray:
    """
    Mean dispatch mass per expert.
    For shapes like [B, M, N, P] (batch, tokens, experts, slots),
    returns [N] = mean over (B, M, P).
    """
    assert dispatch.ndim >= 3, "dispatch must be [..., experts, ...]"
    # average over all dims except expert dim= -2 if shape [B,M,N,P], or infer N index
    # Assuming layout [B, M, N, P] (as in your 'ours' path):
    if dispatch.ndim == 4:
        # mean over B(0), M(1), P(3)
        mass_per_expert = jnp.mean(dispatch, axis=(0, 1, 3))  # [N]
    else:
        # generic: bring expert axis to 0 and mean the rest
        expert_axis = -2
        perm = list(range(dispatch.ndim))
        perm[expert_axis], perm[0] = perm[0], perm[expert_axis]
        d = jnp.transpose(dispatch, perm)  # [N, ...]
        mass_per_expert = jnp.mean(d, axis=tuple(range(1, d.ndim)))  # [N]
    return mass_per_expert


def expert_dormant_fraction(mass_per_expert: jnp.ndarray, tau_exp: float = 1e-3) -> jnp.ndarray:
    """Fraction of experts whose mean dispatch mass is below tau_exp."""
    return jnp.mean(mass_per_expert < tau_exp)


def phi_norms(phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Norms for phi in shape [D, N, P]:
      - global ||phi||_2
      - per-expert ||phi[:, i, :]||_2  -> [N]
      - per-slot   ||phi[:, :, p]||_2  -> [P]
    """
    assert phi.ndim == 3, "phi must be [D, N, P]"
    global_norm = l2(phi)
    per_expert = jnp.linalg.norm(phi, axis=(0, 2))  # [N]
    per_slot = jnp.linalg.norm(phi, axis=(0, 1))  # [P]
    return global_norm, per_expert, per_slot


# --------------------------------------------------------------------------------------
# Q-value diagnostics
# --------------------------------------------------------------------------------------

def q_stats(q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute simple diagnostics on Q-values.
    q: [B, A] or [B, N, A]
      - qnorm: mean L2 norm per sample
      - qvar_actions: mean variance across actions (per-sample), averaged over batch (and experts if present)
      - qvar_batch: variance across batch of the mean Q per sample (and experts averaged if present)
    """
    if q.ndim == 3:  # [B, N, A]
        qn = jnp.mean(jnp.linalg.norm(q, axis=-1))   # scalar
        qvar_actions = jnp.mean(jnp.var(q, axis=-1))  # scalar
        qvar_batch = jnp.var(jnp.mean(q, axis=-1))  # scalar
        return qn, qvar_actions, qvar_batch
    else:            # [B, A]
        qn = jnp.mean(jnp.linalg.norm(q, axis=-1))
        qvar_actions = jnp.mean(jnp.var(q, axis=-1))
        qvar_batch = jnp.var(jnp.mean(q, axis=-1))
        return qn, qvar_actions, qvar_batch


# --------------------------------------------------------------------------------------
# Parameter norms
# --------------------------------------------------------------------------------------

def param_norms(params: Any) -> Dict[str, jnp.ndarray]:
    """
    Per-parameter L2 norms of a parameter PyTree (flattened keys).
    Returns a dict { 'module/.../param': ||leaf||_2 }.
    """
    flat = _flatten_params(params)
    return {k: jnp.linalg.norm(v) for k, v in flat.items()}


def total_param_norm(params: Any) -> jnp.ndarray:
    """Global L2 norm across the entire parameter tree."""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.sqrt(jnp.sum(jnp.array([jnp.sum(jnp.square(x)) for x in leaves])))


# --------------------------------------------------------------------------------------
# Optional: head-only NTK mask convenience
# --------------------------------------------------------------------------------------

def make_head_only_mask(head_key_substr: str = 'action_head') -> Callable[[str], bool]:
    """
    Returns a predicate that is True only for parameters whose flattened key contains `head_key_substr`.
    Use with ntk_srank(..., param_mask_fn=make_head_only_mask('action_head')) to restrict NTK to the head.
    """
    def _pred(k: str) -> bool:
        return head_key_substr in k
    return _pred
