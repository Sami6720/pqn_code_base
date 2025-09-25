"""
This script uses BatchRenorm for more effective batch normalization in long training runs.
"""

import copy
import os
import time
import jax
import flax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.utils.craftax_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)
from purejaxql.utils.batch_renorm import BatchRenorm

from analysis_helpers import (
    effective_rank, effective_rank_per_expert, ntk_srank, dormant_fraction,
    routing_utilization, expert_dormant_fraction, phi_norms, q_stats,
)
import flax
from einops import rearrange, reduce, einsum, repeat

from flax import traverse_util

def count_params(params: Any) -> int:
    """Total number of scalars in a JAX/Flax params PyTree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

class QNetworkPerm(nn.Module):
    action_dim: int
    config: dict
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):

        # >>> CHANGE: enable/disable instrumentation from config
        log_int = self.config.get("LOG_INTERNALS", True)  # when True we sow() intermediates

        if self.norm_type == "layer_norm":
            def normalize(x): return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            def normalize(x): return BatchRenorm(
                use_running_average=not train)(x)
        else:
            def normalize(x): return x

        if "Pixels" in self.config["ENV_NAME"]:
            B, H, W, C = x.shape

            # dummy normalize input for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

            if self.config["FEATURES_FROM_PIXELS_STRAT"] == 'conv':
                initializer = nn.initializers.xavier_uniform()

                x = x.astype(jnp.float32) / 255.0
                x = nn.Conv(
                    features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
                )(x)
                x = nn.relu(x)
                x = nn.Conv(
                    features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
                )(x)
                x = nn.relu(x)
                x = nn.Conv(
                    features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
                )(x)
                x = nn.relu(x)

                # >>> CHANGE: expose CNN output (for dormant-neuron on encoder)
                if log_int:
                    self.sow('intermediates', 'cnn_out', x)


            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_pixels':
                x = x.astype(jnp.float32) / 255.0
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'flattened':
                x = x.reshape(B, -1)
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_flattened':
                x = x.astype(jnp.float32) / 255.0
                x = x.reshape(B, -1)
            else:
                raise ValueError("Wrong way to generate feature from pixels")


            if self.config['USE_SOFT_MOE_MULTI_EXPERT']:

                assert 'flattened' not in self.config["FEATURES_FROM_PIXELS_STRAT"]

                B, H, W, D = x.shape
                #TOKENIZE PerConv
                x = x.reshape(B, -1, D) # Shape (H*W) X D
                if self.config["SOFT_MOE_APPR"] == 'ours':
                    print("Debug print x.shape after conv", x.shape)
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)
                    self.sow("intermediates", "phi_norm", jnp.linalg.norm(phi))

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine_per_expert = jax.nn.softmax(logits, axis=-1)

                    # >>> CHANGE: expose MoE internals for permanent net
                    if log_int:
                        self.sow('intermediates', 'perm_logits', logits)
                        self.sow('intermediates', 'perm_dispatch', dispatch)
                        self.sow('intermediates', 'perm_combine_per_expert', combine_per_expert)

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for j in range(self.num_layers):
                            if j == (self.num_layers - 1):
                                expert_out = nn.Dense(D)(expert_out)
                            else:
                                expert_out = nn.Dense(int(self.hidden_size * 0.88))(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)

                            if log_int:
                                # NEW: expose per-expert, per-layer activations
                                self.sow('intermediates', f'perm_exp{i}_layer{j}_act', expert_out)

                        stack.append(expert_out)

                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD
                    y_tilda_tilda = jnp.einsum('bnpd,bmnp->bnmd', y_tilda, combine_per_expert)
                    # >>> CHANGE: expose per-expert features (for per-expert effective rank)
                    if log_int:
                        self.sow('intermediates', 'perm_y_tilda_tilda', y_tilda_tilda)

                    stack = []
                    # Output layer: one Q-value per action per expert
                    for i in range(self.config["NUM_EXPERTS"]):
                        # The input is a vector of shape M * D into the Q-value head
                        expert_perm_q_val = nn.Dense(self.action_dim)(y_tilda_tilda[:, i, :, :].reshape(B, -1))
                        stack.append(expert_perm_q_val)
                    y = jnp.stack(stack, axis=1) # BNA

                    # >>> CHANGE: expose per-expert Qs
                    if log_int:
                        self.sow('intermediates', 'perm_qs_per_expert', y)

                    if self.config["EXPERT_OUTPUT_COMBINE_STRAT"] == 'sum':
                        x = jnp.sum(y, axis=1)
                        return x
                    elif self.config["EXPERT_OUTPUT_COMBINE_STRAT"] == "softmax_over_n_meanpool":
                        combine_q_vec_temp = self.config.get('MEANPOOL_TEMP', 3e-5)
                        combine_q_vectors = jax.nn.softmax(jnp.mean(logits, axis=(1, 3))/combine_q_vec_temp, axis=1) # BN
                        if log_int:
                            softmax_input = jnp.mean(logits, axis=(1, 3))
                            self.sow('intermediates', 'softmax_input', softmax_input)
                        if log_int:
                            self.sow('intermediates', 'combine_weight_per_expert', combine_q_vectors)
                        x = jnp.einsum("bn,bna->ba", combine_q_vectors, y)
                        return x
                    elif self.config["EXPERT_OUTPUT_COMBINE_STRAT"] == "attn":
                        # y: [B, N, A]  (per-expert Q vectors)
                        dk = int(self.config.get("ATTN_DK", max(32, y.shape[-1] // 2)))
                        temp = float(self.config.get("ATTN_TEMPERATURE", 1.0))
                        stop_g = bool(self.config.get("ATTN_STOP_GRAD", True))

                        # Optionally detach attention inputs (keeps head stable)
                        y_for_attn = jax.lax.stop_gradient(y) if stop_g else y

                        # Keys per expert from its Q vector: [B, N, dk]
                        keys = nn.Dense(dk, name="perm_attn_keys")(y_for_attn)

                        # Global query from mean Q over experts: [B, dk]
                        query = nn.Dense(dk, name="perm_attn_query")(jnp.mean(y_for_attn, axis=1))

                        # Scores: [B, N], scaled dot-product attention
                        scores = jnp.einsum("bnd,bd->bn", keys, query) / jnp.sqrt(dk)
                        scores = (scores - jnp.max(scores, axis=1, keepdims=True)) / temp
                        alpha = jax.nn.softmax(scores, axis=1)  # [B, N]

                        if log_int:
                            self.sow('intermediates', 'combine_weight_per_expert', alpha)

                        # Mix experts by attention weights: [B, A]
                        x = jnp.einsum("bna,bn->ba", y, alpha)
                        return x
                    else:
                        raise ValueError("Incorrect EXPERT_OUTPUT_COMBINE_STRAT")
                elif self.config['SOFT_MOE_APPR'] == 'big':
                #NOTE: Big arch since in Mixture of Experts in Mixtures of RL this it worked the best.
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)

                    self.sow("intermediates", "phi_norm", jnp.linalg.norm(phi))

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine = jax.nn.softmax(logits, axis=(2, 3))

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for j in range(self.num_layers):
                            if j == (self.num_layers - 1):
                                expert_out = nn.Dense(D)(expert_out)
                            else:
                                expert_out = nn.Dense(int(self.hidden_size * 0.88))(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)

                            if log_int:
                                # NEW: expose per-expert, per-layer activations
                                self.sow('intermediates', f'perm_exp{i}_layer{j}_act', expert_out)

                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    x = jnp.einsum("bnpd,bmnp->bmd", y_tilda, combine)
                    x = nn.Dense(self.action_dim)(x.reshape(B, -1))
                    return x
                else:
                    raise ValueError("Incorrect SOFT_MOE_APPR: ", self.config["SOFT_MOE_APPR"])

            else:
                # Flatten the output from encoder if not using soft-moe.
                x = x.reshape(B, -1)
        else:
            if self.norm_input:
                x = BatchRenorm(use_running_average=not train)(x)
            else:
                # dummy normalize input for global compatibility
                x_dummy = BatchRenorm(use_running_average=not train)(x)


            if self.config["USE_TOPK_MULTI_EXPERT"]:

                expert_embeds = self.param("expert_ids", nn.linear.default_embed_init, (self.config["NUM_EXPERTS"], self.hidden_size))
                input_expert_query = nn.Dense(self.hidden_size)(x)
                scores = einsum(expert_embeds, input_expert_query, "n d, b d -> b n")
                # probs = nn.softmax(scores, axis=-1)

                topk_scores, topk_idx = jax.lax.top_k(scores, self.config["TOPK"]) # (B, K), (B, K)

                #NOTE: aux loss computation.
                # Hard usage f_i (non-diff): count how often each expert was selected in top-k
                alpha = float(self.config.get("EXP_BAL_ALPHA", 0.001))
                B, N = scores.shape
                K = topk_idx.shape[-1]
                counts = jnp.bincount(
                    topk_idx.reshape(-1), minlength=N)
                f_i = counts.astype(jnp.float32) / (B * K)
                # Soft marginal P_i (diff): average full softmax over all experts
                probs_all = nn.softmax(scores, axis=-1)                                  # [B, N]
                P_i = jnp.mean(probs_all, axis=0)                                        # [N]
                aux_loss = alpha * jnp.sum(P_i, f_i)
                self.sow("load_balancing", "aux_loss", aux_loss)

                # mask = jnp.zeros_like(scores)
                # mask = mask.at[jnp.arange(scores.shape[0])[:, None], topk_idx].set(1)
                # scores *= mask
                # gates = topk_scores / topk_scores.sum(axis=-1, keepdims=True)
                gates = nn.softmax(topk_scores, axis=-1)


                for i in range(self.num_layers):
                    B, D = x.shape

                    W = self.param(f"layer_{i}_kernel", nn.linear.default_kernel_init,
                                   (self.config["NUM_EXPERTS"],
                                   D, self.hidden_size))
                    b = self.param(f"layer_{i}_bias", nn.initializers.zeros_init, (self.config['NUM_EXPERTS'], self.hidden_size))
                    W_sel = W[topk_idx]
                    b_sel = b[topk_idx]
                    x = einsum(x, W_sel, "b d, b k d d_out -> b k d_out")
                    x += b_sel

                    x = normalize(x)
                    x = nn.relu(x)


                x = einsum(x, gates, "b k d, b k -> b d")

                x = nn.Dense(self.action_dim)(x)
                return x


        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

            if log_int:
                self.sow('intermediates', f'perm_layer{l}_act', x)

        # >>> CHANGE: expose last hidden before action head (even if you won't use full-module feature rank)
        if log_int:
            self.sow('intermediates', 'last_hidden', x)

        x = nn.Dense(self.action_dim)(x)

        return x


class QNetwork(nn.Module):
    action_dim: int
    config: dict
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):

        log_int = self.config.get("LOG_INTERNALS", True)  # when True we sow() intermediates

        if self.norm_type == "layer_norm":
            def normalize(x): return nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            def normalize(x): return BatchRenorm(
                use_running_average=not train)(x)
        else:
            def normalize(x): return x

        if "Pixels" in self.config["ENV_NAME"]:
            B, H, W, C = x.shape

            # dummy normalize input for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

            if self.config["FEATURES_FROM_PIXELS_STRAT"] == 'conv':
                initializer = nn.initializers.xavier_uniform()

                x = x.astype(jnp.float32) / 255.0
                x = nn.Conv(
                    features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer
                )(x)
                x = nn.relu(x)
                x = nn.Conv(
                    features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer
                )(x)
                x = nn.relu(x)
                x = nn.Conv(
                    features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer
                )(x)
                x = nn.relu(x)

                # >>> CHANGE: expose CNN output (for dormant-neuron on encoder)
                if log_int:
                    self.sow('intermediates', 'cnn_out', x)

            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_pixels':
                x = x.astype(jnp.float32) / 255.0
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'flattened':
                x = x.reshape(B, -1)
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_flattened':
                x = x.astype(jnp.float32) / 255.0
                x = x.reshape(B, -1)
            else:
                raise ValueError("Wrong way to generate feature from pixels")


            if self.config['USE_SOFT_MOE_MULTI_EXPERT_TRANS']:

                assert 'flattened' not in self.config["FEATURES_FROM_PIXELS_STRAT"]

                B, H, W, D = x.shape
                print("Debug print x.shape after conv", x.shape)
                #TOKENIZE PerConv
                x = x.reshape(B, -1, D) # Shape (H*W) X D

                if self.config["SOFT_MOE_APPR_TRANS"] == 'ours':
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine_per_expert = jax.nn.softmax(logits, axis=-1)

                    # >>> CHANGE: expose MoE internals for permanent net
                    if log_int:
                        self.sow('intermediates', 'trans_logits', logits)
                        self.sow('intermediates', 'trans_dispatch', dispatch)
                        self.sow('intermediates', 'trans_combine_per_expert', combine_per_expert)

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for j in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    stack = []
                    y_tilda_tilda = jnp.einsum('bnpd,bmnp->bnmd', y_tilda, combine_per_expert)
                    # Output layer: one Q-value per action per expert

                    # >>> CHANGE: expose per-expert features (for per-expert effective rank)
                    if log_int:
                        self.sow('intermediates', 'trans_y_tilda_tilda', y_tilda_tilda)

                    for i in range(self.config["NUM_EXPERTS"]):
                        # The input is a vector of shape M * D into the Q-value head
                        expert_perm_q_val = nn.Dense(self.action_dim)(y_tilda_tilda[:, i, :, :].reshape(B, -1))
                        stack.append(expert_perm_q_val)
                    y = jnp.stack(stack, axis=1) # BNA

                    if self.config["EXPERT_OUTPUT_COMBINE_STRAT_TRANS"] == 'sum':
                        x = jnp.sum(y, axis=1)
                        return x
                    elif self.config["EXPERT_OUTPUT_COMBINE_STRAT_TRANS"] == "softmax_over_n_meanpool":
                        combine_q_vectors = jax.nn.softmax(jnp.mean(logits, axis=(1, 3)), axis=1) # BN
                        x = jnp.einsum("bn,bna->ba", combine_q_vectors, y)
                        return x
                    else:
                        raise ValueError("Incorrect EXPERT_OUTPUT_COMBINE_STRAT")
                elif self.config['SOFT_MOE_APPR_TRANS'] == 'big':
                #NOTE: Big arch since in Mixture of Experts in Mixtures of RL this it worked the best.
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine = jax.nn.softmax(logits, axis=(2, 3))

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for j in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    x = jnp.einsum("bnpd,bmnp->bmd", y_tilda, combine)
                    x = nn.Dense(self.action_dim)(x.reshape(B, -1))
                    return x
            else:
                # Flatten the output from encoder if not using soft-moe.
                x = x.reshape(B, -1)
        else:
            if self.norm_input:
                x = BatchRenorm(use_running_average=not train)(x)
            else:
                # dummy normalize input for global compatibility
                x_dummy = BatchRenorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

            # NEW: expose per-layer activations
            if log_int:  # already defined in your code
                self.sow('intermediates', f'trans_layer{l}_act', x)

        # >>> CHANGE: expose last hidden before action head (even if you won't use full-module feature rank)
        if log_int:
            self.sow('intermediates', 'last_hidden', x)


        x = nn.Dense(self.action_dim, name='action_head')(x)

        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array
    old_p_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions


    from einops import rearrange


    #NOTE: Observations for computing dormant neurons
    all_obs_files = os.listdir("obs_fixed")[:8]
    all_processed_obs = [rearrange(jnp.load(f'obs_fixed/{f}'), 'x b ... -> (x b) ...') for f in all_obs_files]
    all_processed_obs = jnp.stack(all_processed_obs, axis=0)
    all_processed_obs = rearrange(all_processed_obs, 'x b ... -> (x b) ...')
    print(f"All processed observation shapes {all_processed_obs.shape}")


    def train(rng):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            config=config
        )

        network_perm = QNetworkPerm(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            config=config
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state


        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        transient_parameters_count = count_params(train_state.params)

        print(f"Transient Network params count: {transient_parameters_count}")

        if config["USE_PERM"]:
            def create_agent_perm(rng):
                init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
                network_variables = network_perm.init(rng, init_x, train=False)
                lr_scheduler = optax.linear_schedule(
                    init_value=config["LR_PERM"],
                    end_value=1e-21,
                    #TODO: Maybe transition_steps needs to change?
                    transition_steps=(config["NUM_UPDATES_DECAY"])
                    * config["NUM_MINIBATCHES"]
                    * config["NUM_EPOCHS"],
                )
                lr = lr_scheduler if config.get("LR_PERM_LINEAR_DECAY", False) else config["LR_PERM"]
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.radam(learning_rate=lr),
                )

                train_state = CustomTrainState.create(
                    apply_fn=network_perm.apply,
                    params=network_variables["params"],
                    batch_stats=network_variables["batch_stats"],
                    tx=tx,
                )
                return train_state
            rng, _rng = jax.random.split(rng)
            train_state_perm = create_agent_perm(_rng)
            permanent_network_parameter_count = count_params(train_state_perm.params)
            print(f"Permanent Network params count: {permanent_network_parameter_count}")
            print("Hidden size: ", network_perm.hidden_size)
        else:
            permanent_network_parameter_count = 0
            train_state_perm = None

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, train_state_perm, expl_state, test_metrics, rng = runner_state

            old_params_perm = train_state_perm.params

            metrics = {}

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                #TODO: Add q_perm values to the q_vals here.
                if config["USE_PERM"]:
                    print("Get here")
                    q_vals_perm = network_perm.apply(
                        {
                            "params": train_state_perm.params,
                            "batch_stats": train_state_perm.batch_stats
                        },
                        last_obs,
                        train=False
                    )

                    q_vals += q_vals_perm
                else:
                    q_vals_perm = jnp.zeros((config["NUM_ENVS"], env.action_space(env_params).n))

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                if config["USE_PERM"]:
                    # Q from perm and total at the chosen action
                    q_perm_sel  = jnp.take_along_axis(q_vals_perm, jnp.expand_dims(new_action, -1), axis=-1).squeeze(-1)  # [NUM_ENVS]
                    q_total_sel = jnp.take_along_axis(q_vals, jnp.expand_dims(new_action, -1), axis=-1).squeeze(-1)  # [NUM_ENVS]

                    eps_den = jnp.asarray(1e-8, q_total_sel.dtype)

                    q_trans_sel = q_total_sel - q_perm_sel
                    q_val_perm_proportion = jnp.abs(q_perm_sel) / (jnp.abs(q_perm_sel) + jnp.abs(q_trans_sel) + eps_den)
                else:
                    q_val_perm_proportion = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                    old_p_val=q_vals_perm
                )
                return (new_obs, new_env_state, rng), (transition, info, q_val_perm_proportion)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos, q_val_perm_proportions) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # def save_callback(observations, timesteps):
            #     import numpy as np
            #     is_save_time = timesteps % 24999936 == 0
            #     # is_save_time = timesteps % 1
            #     if is_save_time:
            #         np.save(f"obs_{timesteps}.npz", np.array(observations))
            #
            # jax.debug.callback(save_callback, transitions.obs, train_state.timesteps)
            #

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                #TODO: Why use -1 to index
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            metrics["q_val_perm_proportion"] = jnp.nanmean(q_val_perm_proportions)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params, params_perm):

                        if config.get("Q_LAMBDA", False):
                            q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                        else:
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)

                            #TODO: Need to add q_perm to both. Like an all_q_vals_perm
                            if config["USE_PERM"]:
                                all_q_vals_perm = network_perm.apply(
                                    {
                                        "params": params_perm,
                                        "batch_stats": train_state_perm.batch_stats,
                                    },
                                    jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                    train=False,
                                    )
                                q_vals_perm, q_next_perm = jnp.split(all_q_vals_perm, 2)
                                q_vals += q_vals_perm
                                q_next += q_next_perm


                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            # NOTE: Lambda target from above is overwritten here when Q_Lamda is False
                            target = (
                                minibatch.reward
                                + (1 - minibatch.done) * config["GAMMA"] * q_next
                            )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    if config["USE_PERM"]:
                        perm_params = train_state_perm.params
                    else:
                        perm_params = None
                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params, perm_params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals, grads)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals, grads) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals, jax.flatten_util.ravel_pytree(grads)[0])

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals, grads) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            print("transient grads shape", grads.shape)
            metrics_ = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "trans/grad_norm": jnp.mean(jnp.linalg.norm(grads, axis=-1))
            }

            metrics['perm_parameter_count'] = permanent_network_parameter_count
            metrics['trans_parameter_count'] = transient_parameters_count

            metrics.update(metrics_)
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            metrics.update(done_infos)

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            # remove achievement metrics if not logging them
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            #TODO: Train the perm network here over the same minibatches with a jax.lax.cond
            # UPDATE Perm Network
            print("Get above _learn_epoch_perm")
            if config["USE_PERM"]:
                def _learn_epoch_perm(carry, _):
                    train_state_perm, train_state, rng = carry
                    def _learn_phase_perm(carry, minibatch_and_target):

                        train_state_perm, rng = carry
                        minibatch, _ = minibatch_and_target

                        def _loss_fn_perm(params_perm, params_trans):

                            if config["USE_TOPK_MULTI_EXPERT"]:
                                q_vals_perm, updates_perm = network_perm.apply(
                                        {
                                        "params": params_perm, 
                                        "batch_stats": train_state_perm.batch_stats
                                        },
                                    minibatch.obs,
                                    mutable=["batch_stats", "load_balancing"],
                                    train=True
                                )
                            else:
                                q_vals_perm, updates_perm = network_perm.apply(
                                        {
                                        "params": params_perm, 
                                        "batch_stats": train_state_perm.batch_stats
                                        },
                                    minibatch.obs,
                                    mutable=["batch_stats"],
                                    train=True
                                )
                            q_vals_perm = jnp.take_along_axis(
                                q_vals_perm,
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            q_vals_trans = network.apply(
                                    {
                                    "params": params_trans,
                                    "batch_stats": train_state.batch_stats
                                    },
                                minibatch.obs,
                                train=False
                            )
                            q_vals_trans = jnp.take_along_axis(
                                q_vals_trans,
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            old_p_val = jnp.take_along_axis(
                                minibatch.old_p_val,
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)

                            target = jax.lax.stop_gradient(q_vals_trans + old_p_val)

                            loss = 0.5 * jnp.square(target - q_vals_perm).mean()

                            if config["USE_TOPK_MULTI_EXPERT"]:
                                loss += updates_perm["load_balancing"]["aux_loss"][-1]

                            return loss, updates_perm

                        (loss_perm, updates_perm), grads = jax.value_and_grad(
                            _loss_fn_perm, has_aux=True
                        )(train_state_perm.params, train_state.params)
                        train_state_perm = train_state_perm.apply_gradients(grads=grads)
                        train_state_perm = train_state_perm.replace(
                            grad_steps=train_state_perm.grad_steps + 1,
                            batch_stats=updates_perm["batch_stats"],
                        )

                        import flax
                        # Flatten grads to find phi
                        if config["USE_SOFT_MOE_MULTI_EXPERT"]:
                            flat_grads = flax.traverse_util.flatten_dict(grads, sep="/")
                            phi_key = [k for k in flat_grads.keys() if "phi" in k][0]
                            phi_grad = flat_grads[phi_key]
                            print("Phi grad shape ", phi_grad.shape)
                            # phi_grad_norm = jnp.linalg.norm(phi_grad)
                            # print(f"Phi grad shape {phi_grad.shape}")
                            # else jnp.full(shape_of_phi_grad)
                        else:
                            phi_grad = jnp.full((64, 4, 16), jnp.nan)
                        # returns loss_perm, grads, grads_phi

                        return (train_state_perm, rng), (loss_perm, grads, phi_grad)

                    def preprocess_transition_perm(x, rng):
                        x = x.reshape(
                            -1, *x.shape[2:]
                        )  # num_steps*num_envs (batch_size), ...
                        x = jax.random.permutation(rng, x)  # shuffle the transitions
                        x = x.reshape(
                            config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                        )  # num_mini_updates, batch_size/num_mini_updates, ...
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree_util.tree_map(
                        lambda x: preprocess_transition_perm(x, _rng), transitions
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree_util.tree_map(
                        lambda x: preprocess_transition_perm(x, _rng), lambda_targets
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state_perm, rng), (loss, grads_perm, phi_grads) = jax.lax.scan(
                        _learn_phase_perm, (train_state_perm, rng), (minibatches, targets)
                    )

                    # Soft-reset transient. Keeping the transient-weight the same across the minibatches so only resetting at end of epoch.
                    def reset_transient(train_state_trans, rng):


                        if config['TRANS_WEIGHT_RESET_STRATEGY'] == 'exp':
                            temp_train_state = train_state_trans.replace(
                                params=jax.tree_map(
                                    lambda x: (config["TRANSIENT_WEIGHT_DECAY"] ** (train_state_perm.n_updates)) *  x, train_state_trans.params
                                ))
                        elif config["TRANS_WEIGHT_RESET_STRATEGY"] == 'multiplicative':
                            temp_train_state = train_state_trans.replace(
                                params=jax.tree_map(
                                    lambda x: config["TRANSIENT_WEIGHT_DECAY"] *  x, train_state_trans.params
                                )
                            )
                        elif config["TRANS_WEIGHT_RESET_STRATEGY"] == 'no_reset':
                            temp_train_state = train_state_trans
                        elif config["TRANS_WEIGHT_RESET_STRATEGY"] == 'reinit_action_heads':
                            assert config["USE_SOFT_MOE_MULTI_EXPERT_TRANS"] == False 
                            rng, _rng = jax.random.split(rng)
                            def reinit_action_heads(ts, rng, dummy_input):
                                # Get a fresh param tree for this module on the same input shape
                                new_params_full = network.init(rng, dummy_input, train=False)['params']
                                # Splice only the head(s)
                                import flax
                                flat_old = flax.traverse_util.flatten_dict(ts.params, sep="/")
                                flat_new = flax.traverse_util.flatten_dict(new_params_full, sep="/")
                                # jax.debug.print("get here, {k}", k=flat_new.keys())
                                # jax.debug.print("get here, old {k}", k=flat_old.keys())
                                # print(f"get here, {flat_new.keys()}")
                                # print(f"get here, old {flat_old.keys()}")
                                def is_head(k):
                                    return 'action_head' in k
                                for k in list(flat_old.keys()):
                                    if is_head(k):
                                        flat_old[k] = flat_new[k]


                                return ts.replace(params=flax.traverse_util.unflatten_dict(flat_old, sep="/"))

                            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
                            temp_train_state = reinit_action_heads(train_state, _rng, init_x)
                        else:
                            raise ValueError("Wrong TRANS_WEIGHT_RESET_STRATEGY: ", config["TRANS_WEIGHT_RESET_STRATEGY"])

                        return temp_train_state, rng

                    should_reset = (train_state_perm.n_updates % config["TRANS_RESET_FREQ_EVERY_PERM_UPDATE"] == 0)
                    modified_train_states_trans, rng = jax.lax.cond(
                        should_reset,
                        reset_transient,
                        lambda train_states_trans, rng: (train_states_trans, rng),
                       train_state,
                        rng
                    )


                    import flax
                    from flax import traverse_util

                    print(type(grads_perm))
                    print(jax.flatten_util.ravel_pytree(grads_perm)[0].shape)


                    print(jax.flatten_util.ravel_pytree(phi_grads)[0].shape)

                    return (train_state_perm, modified_train_states_trans, rng), (loss, jax.flatten_util.ravel_pytree(grads_perm)[0],
                                                                                  jax.flatten_util.ravel_pytree(phi_grads)[0]
                                                                                  )


                rng, _rng = jax.random.split(rng)
                is_perm_learn_time = (
                    train_state.n_updates % config["PERM_UPDATE_FREQ"] == 0
                )

                # None instead of zero since wandb filters out automatically.
                dummy_loss = jnp.full(
                    (config["NUM_EPOCHS_PERM"], config["NUM_MINIBATCHES"]), jnp.nan)
                dummy_grad = jnp.full(
                    (config["NUM_EPOCHS_PERM"], permanent_network_parameter_count), jnp.nan)
                dummy_phi_grad = jnp.full(
                    (config["NUM_EPOCHS_PERM"], 4096), jnp.nan # 4096 is the number of parameters in phi if use_soft_moe
                )
                (train_state_perm, train_state, rng), (loss_perm, grad_perm, phi_grad) = jax.lax.cond(
                    is_perm_learn_time,
                    lambda train_state_perm, train_state, rng: jax.lax.scan(
                        _learn_epoch_perm, (train_state_perm, train_state,
                                            rng), None, config["NUM_EPOCHS_PERM"]
                    ),
                    lambda train_state_perm, train_state, rng: ((train_state_perm, train_state, rng), (dummy_loss, dummy_grad, dummy_phi_grad)),
                    train_state_perm,
                    train_state,
                    _rng
                )
                train_state_perm = jax.lax.cond(
                    is_perm_learn_time,
                    lambda ts: ts.replace(n_updates=ts.n_updates + 1),
                    lambda ts: ts,
                    train_state_perm
                )

                metrics["perm/loss"] = jnp.nanmean(loss_perm)
                print("permanent gradient shape", grad_perm.shape)
                print("phi gradient shape", phi_grad.shape)
                assert len(grad_perm.shape) == 2
                assert len(phi_grad.shape) == 2
                metrics["perm/grad_norm"] = jnp.nanmean(jnp.linalg.norm(grad_perm, axis=-1))
                metrics["perm/phi_grad_norm"] = jnp.nanmean(jnp.linalg.norm(phi_grad, axis=-1))

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                # ===== Analysis knobs =====
                ANALYSIS_INTERVAL = int(config.get("WANDB_LOG_INTERVAL", 10))
                ANALYSIS_BATCH    = int(config.get("ANALYSIS_BATCH", 256))
                SRANK_DELTA       = float(config.get("SRANK_DELTA", 0.01))
                # NOTE: Tau chosen from the Dormant Neuron Phenomenon in DRL
                DORMANT_TAU       = float(config.get("DORMANT_TAU", 0.025))
                LOG_INTERNALS     = bool(config.get("LOG_INTERNALS", True))

                import flax
                from flax import traverse_util

                # ----- util: ensure every metric is float32 to satisfy lax.cond type equality -----
                def _f32(x):
                    return jnp.asarray(x, jnp.float32)

                # ----- util: param mask for "exclude CNN" NTK -----
                def _mask_excl_cnn(key: str) -> bool:
                    return ("Conv" not in key)

                # ----- util: sample a small batch from transitions for analysis -----
                def _sample_obs_for_analysis(transitions, rng):
                    obs = transitions.obs.reshape(-1, *transitions.obs.shape[2:])
                    n = obs.shape[0]
                    if n > ANALYSIS_BATCH:
                        idx = jax.random.choice(rng, n, (ANALYSIS_BATCH,), replace=False)
                        obs = obs[idx]
                    return obs

                # ----- SAFE accessors for intermediates (handle missing keys & tuples) -----
                def _get_intermediates(coll):
                    if isinstance(coll, (dict, flax.core.FrozenDict)):
                        return coll.get('intermediates', {})
                    return {}

                def _as_array(v):
                    if isinstance(v, (list, tuple)):
                        if not v:
                            return None
                        v = v[-1]
                    try:
                        return jnp.asarray(v)
                    except Exception:
                        return None

                def _last_sown(inter_dict, name: str):
                    if not isinstance(inter_dict, (dict, flax.core.FrozenDict)):
                        return None
                    flat = traverse_util.flatten_dict(inter_dict, sep='/')  # "Module/.../name"
                    candidates = []
                    for k, v in flat.items():
                        if isinstance(k, str) and (k.endswith('/' + name) or k == name):
                            vv = _as_array(v)
                            if vv is not None:
                                candidates.append(vv)
                    if not candidates:
                        return None
                    return candidates[-1]

                # ----- build NaN metrics WITHOUT doing heavy work in the false branch -----
                def _empty_analysis_metrics():
                    nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
                    out = {
                        # transient
                        "trans/feat_srank": nan,
                        "trans/qnorm": nan,
                        "trans/dormant_all": nan,
                        "trans/dormant_all_fixed": nan,
                        "trans/qvar_actions": nan,
                        "trans/qvar_batch": nan,
                        "trans/param_norm": nan,

                        # permanent (overall)
                        "perm/qnorm": nan,
                        "perm/qvar_actions": nan,
                        "perm/qvar_batch": nan,
                        "perm/param_norm": nan,
                        "perm/grad_steps": nan,
                        "perm/n_updates": nan,
                        "perm/update_l2": nan,
                        "perm/phi_norm": nan,
                        #Not soft-moe
                        "perm/dormant_all": nan,
                        "perm/dormant_all_fixed": nan,
                        "perm/softmax_input_norm":nan,
                        "perm/feat_srank": nan
                    }
                    # NEW: per-expert dormant placeholders (MLP-only)
                    for i in range(int(config.get("NUM_EXPERTS", 1))):
                        out[f"perm/expert_{i}/dormant_all"] = nan
                        out[f"perm/expert_{i}/dormant_all_fixed"] = nan
                        out[f"perm/expert_{i}/feat_srank"] = nan
                        out[f"perm/expert_{i}/qnorm"] = nan
                        out[f"perm/expert_{i}/qvar_actions"] = nan
                        out[f"perm/expert_{i}/qvar_batch"] = nan
                        out[f"perm/expert_{i}/weight"] = nan

                    return out

                # ----- Core analysis (heavy path runs only at interval) -----
                def _compute_analysis(train_state, train_state_perm, transitions, rng):

                    from typing import Optional

                    def _per_unit_means(acts: jnp.ndarray) -> jnp.ndarray:
                        # one mean per unit/channel, averaged over batch & spatial/tokens

                        per_neuron_mean = reduce(jnp.abs(acts), "... hidden_dim -> hidden_dim", "mean")
                        layer_mean = reduce(per_neuron_mean, "h -> 1", "mean")

                        per_unit = per_neuron_mean/layer_mean

                        assert len(per_unit.shape) == 1
                        return per_unit  # [units]

                    def _concat_means(arrs: list) -> Optional[jnp.ndarray]:
                        """
                        arrs: shape is L, B, D
                        """
                        vecs = []
                        # Essentially iterating over each layer and then finding the mean
                        # for each neuron over the batch.
                        for a in arrs:
                            # if a is None:
                            #     continue
                            # try:
                            vecs.append(_per_unit_means(a))
                            # except Exception:
                            #     pass
                        # if not vecs:
                        #     return None
                        return jnp.concatenate(vecs)

                    out = _empty_analysis_metrics()  # correct structure & dtypes

                    # sample a small batch
                    rng, rng_idx = jax.random.split(rng)
                    obs = _sample_obs_for_analysis(transitions, rng_idx)

                    # -------- Transient (QNetwork) --------
                    variables_t = {"params": train_state.params, "batch_stats": train_state.batch_stats}
                    if LOG_INTERNALS:
                        q_t, coll_t = network.apply(variables_t, obs, train=False, mutable=['intermediates'])
                        inter_t = _get_intermediates(coll_t)
                        q_t_fixed, coll_t_fixed = network.apply(variables_t, all_processed_obs, train=False, mutable=['intermediates'])
                        inter_t_fixed = _get_intermediates(coll_t_fixed)
                    else:
                        q_t = network.apply(variables_t, obs, train=False)
                        inter_t = {}
                        inter_t_fixed = {}

                    if LOG_INTERNALS:
                        trans_layer_acts = []
                        for j in range(int(config.get("NUM_LAYERS", 2))):
                            a = _last_sown(inter_t, f'trans_layer{j}_act')  # [B, D] (post-ReLU)
                            print(f"a.shape {a.shape}")
                            if a is not None:
                                trans_layer_acts.append(a)
                        means_all = _concat_means(trans_layer_acts) # [L, H]
                        out["trans/dormant_all"] = _f32(jnp.mean(means_all < DORMANT_TAU))
                        trans_layer_acts_fixed = []
                        for j in range(int(config.get("NUM_LAYERS", 2))):
                            a = _last_sown(inter_t_fixed, f'trans_layer{j}_act')  # [B, D] (post-ReLU)
                            if a is not None:
                                trans_layer_acts_fixed.append(a)
                        means_all = _concat_means(trans_layer_acts_fixed) # [L, H]
                        out["trans/dormant_all_fixed"] = _f32(jnp.mean(means_all < DORMANT_TAU))

                    # last_hidden effective rank
                    last_hidden_t = _last_sown(inter_t, 'last_hidden')
                    out["trans/feat_srank"] = _f32(effective_rank(last_hidden_t, SRANK_DELTA))

                    # Q diagnostics
                    qn_t, qvarA_t, qvarB_t = q_stats(q_t)
                    out["trans/qnorm"] = _f32(qn_t)
                    out["trans/qvar_actions"] = _f32(qvarA_t)
                    out["trans/qvar_batch"] = _f32(qvarB_t)

                    # -------- Permanent (QNetworkPerm) --------
                    if config["USE_PERM"]:
                        variables_p = {"params": train_state_perm.params, "batch_stats": train_state_perm.batch_stats}
                        if LOG_INTERNALS:
                            q_p, coll_p = network_perm.apply(variables_p, obs, train=False, mutable=['intermediates'])
                            inter_p = _get_intermediates(coll_p)
                            q_p_fixed, coll_p_fixed = network_perm.apply(variables_p, all_processed_obs, train=False, mutable=['intermediates'])
                            inter_p_fixed = _get_intermediates(coll_p_fixed)
                        else:
                            q_p = network_perm.apply(variables_p, obs, train=False)
                            inter_p = {}
                            inter_p_fixed = {}

                        # --- per-expert MLP-only dormant (skip CNN) ---
                        if config["USE_SOFT_MOE_MULTI_EXPERT"]:
                            if LOG_INTERNALS:
                                num_experts = int(config.get("NUM_EXPERTS", 1))
                                num_layers  = int(config.get("NUM_LAYERS", 2))
                                for i in range(num_experts):
                                    acts_i = []
                                    for j in range(num_layers):
                                        a = _last_sown(inter_p, f'perm_exp{i}_layer{j}_act')  # [B, ...], sowed after each Dense->Norm->ReLU
                                        print(f"a.shape {a.shape}")
                                        #TODO: NEED TO FIGURE OUT WHY len(a.shape) != 2
                                        if a is not None:
                                            acts_i.append(a)
                                    means_i = _concat_means(acts_i)  # 1D: all units across expert’s hidden layers
                                    out[f"perm/expert_{i}/dormant_all"] = _f32(jnp.mean(means_i < DORMANT_TAU))
                                for i in range(num_experts):
                                    acts_i = []
                                    for j in range(num_layers):
                                        a = _last_sown(inter_p_fixed, f'perm_exp{i}_layer{j}_act')  # [B, ...], sowed after each Dense->Norm->ReLU
                                        if a is not None:
                                            acts_i.append(a)
                                    means_i = _concat_means(acts_i)  # 1D: all units across expert’s hidden layers
                                    out[f"perm/expert_{i}/dormant_all_fixed"] = _f32(jnp.mean(means_i < DORMANT_TAU))

                            # per-expert feature srank (NOT full-module rank)
                            y_tilda_tilda = _last_sown(inter_p, 'perm_y_tilda_tilda')  # [B,N,M,D]
                            if y_tilda_tilda is not None:
                                sranks = effective_rank_per_expert(y_tilda_tilda, delta=SRANK_DELTA)  # [N], float
                                # NEW: log per-expert sranks as separate metrics
                                num_experts = int(config.get("NUM_EXPERTS", 1))
                                for i in range(num_experts):
                                    out[f"perm/expert_{i}/feat_srank"] = _f32(sranks[i])
                        else:
                            last_hidden_p = _last_sown(inter_p, 'last_hidden')
                            out["perm/feat_srank"] = _f32(effective_rank(last_hidden_p, SRANK_DELTA))
                            acts_i = []
                            num_layers  = int(config.get("NUM_LAYERS", 2))
                            for j in range(num_layers):
                                a = _last_sown(inter_p, f'perm_layer{j}_act')  # [B, ...], sowed after each Dense->Norm->ReLU
                                if a is not None:
                                    acts_i.append(a)
                            means_i = _concat_means(acts_i)  # 1D: all units across expert’s hidden layers
                            out[f"perm/dormant_all"] = _f32(jnp.mean(means_i < DORMANT_TAU))
                            acts_i = []
                            for j in range(num_layers):
                                a = _last_sown(inter_p_fixed, f'perm_layer{j}_act')  # [B, ...], sowed after each Dense->Norm->ReLU
                                if a is not None:
                                    acts_i.append(a)
                            means_i = _concat_means(acts_i)  # 1D: all units across expert’s hidden layers
                            out[f"perm/dormant_all_fixed"] = _f32(jnp.mean(means_i < DORMANT_TAU))

                        # Q diagnostics (overall permanent)
                        qn_p, qvarA_p, qvarB_p = q_stats(q_p)
                        out["perm/qnorm"] = _f32(qn_p)
                        out["perm/qvar_actions"] = _f32(qvarA_p)
                        out["perm/qvar_batch"] = _f32(qvarB_p)

                        # Per-expert Q diagnostics
                        if config["USE_SOFT_MOE_MULTI_EXPERT"]:

                            if config["SOFT_MOE_APPR"] == 'ours':
                                q_per_exp = _last_sown(inter_p, 'perm_qs_per_expert')  # [B,N,A]
                                B, N, A = q_per_exp.shape
                                for i in range(N):
                                    q_p = q_per_exp[:, i, :]
                                    qn_p, qvarA_p, qvarB_p = q_stats(q_p)
                                    out[f"perm/expert_{i}/qnorm"] = _f32(qn_p)
                                    out[f"perm/expert_{i}/qvar_actions"] = _f32(qvarA_p)
                                    out[f"perm/expert_{i}/qvar_batch"] = _f32(qvarB_p)


                                # Combine weight diagnostics
                                combine_weight_per_expert = _last_sown(inter_p, 'combine_weight_per_expert') # B, N
                                mean_combine_weight_per_expert = combine_weight_per_expert.mean(axis=0).reshape(-1)
                                for i in range(N):
                                    out[f"perm/expert_{i}/weight"] = mean_combine_weight_per_expert[i]
                                softmax_input = _last_sown(inter_p, 'softmax_input') # B, N
                                print("Softmax input shape: ", softmax_input.shape)
                                softmax_input_norm = jnp.linalg.norm(softmax_input, axis=-1)
                                out[f"perm/softmax_input_norm"] = softmax_input_norm.mean()


                            out["perm/phi_norm"] = _last_sown(inter_p, 'phi_norm')


                    # Parameter norm
                    def tree_l2_norm(params):
                        """Compute the L2 norm of all parameters in a PyTree."""
                        leaves, _ = jax.tree_util.tree_flatten(params)
                        return jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in leaves))

                    out["trans/param_norm"] = _f32(tree_l2_norm(train_state.params))
                    out["perm/param_norm"] = _f32(tree_l2_norm(train_state_perm.params))

                    # --- Are we stepping perm at all? ---
                    out["perm/grad_steps"] = _f32(train_state_perm.grad_steps)
                    out["perm/n_updates"]  = _f32(train_state_perm.n_updates)


                    # Logging the norm of the update to the parameter norm
                    upd = jax.tree_util.tree_map(lambda a,b: a-b, train_state_perm.params, old_params_perm)
                    upd_l2 = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree_util.tree_leaves(upd)))
                    out["perm/update_l2"] = _f32(upd_l2)

                    return out

                # ===== Gate heavy analysis sparsely inside the JIT =====
                us = metrics["update_steps"]
                us0 = us if jnp.ndim(us) == 0 else us[0]   # keep predicate scalar
                do_analyze = (us0 % ANALYSIS_INTERVAL) == 0

                analysis_metrics = jax.lax.cond(
                    do_analyze,
                    lambda _: _compute_analysis(
                        train_state, train_state_perm, transitions,
                        jax.random.fold_in(rng, us0)
                    ),
                    lambda _: _empty_analysis_metrics(),
                    operand=None,
                )

                # merge into metrics pytree (structure stays constant across steps)
                metrics = {**metrics, **analysis_metrics}

                def callback(metrics, original_rng):
                    
                    # log at intervals 
                    if (
                        metrics["update_steps"] % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
                        us = metrics["update_steps"]
                        to_log = {k: (float(v) if hasattr(v, "dtype") else v) for k, v in metrics.items()}
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            old_env_steps = metrics["env_step"]
                            metrics = {
                                    f"rng{int(original_rng)}/{k}": v
                                    for k, v in to_log.items()
                                }
                            metrics["env_step"] = old_env_steps
                        wandb.log(metrics, step=us)

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, train_state_perm, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, train_state_perm, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "perm_pqn_craftax")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True
    )

    submodule_path = "pqn_code_base"
    import subprocess
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=submodule_path
    ).decode("utf-8").strip()
    wandb.config.update({"submodule_commit": commit_hash})

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        # return outs

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
