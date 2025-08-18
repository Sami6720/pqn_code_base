"""
This script uses BatchRenorm for more effective batch normalization in long training runs.
"""

import copy
import os
import time
import jax
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

class QNetworkPerm(nn.Module):
    action_dim: int
    config: dict
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):


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
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_pixels':
                x = x.astype(jnp.uint32) / 255.0
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'flattened':
                x = x.reshape(B, -1)
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_flattened':
                x = x.astype(jnp.uint32) / 255.0
                x = x.reshape(B, -1)
            else:
                raise ValueError("Wrong way to generate feature from pixels")


            if self.config['USE_SOFT_MOE_MULTI_EXPERT']:

                assert 'flattened' not in self.config["FEATURES_FROM_PIXELS_STRAT"]

                if self.config["SOFT_MOE_APPR"] == 'ours':
                    B, H, W, D = x.shape
                    print("Debug print x.shape after conv", x.shape)
                    #TOKENIZE PerConv
                    x = x.reshape(B, -1, D) # Shape (H*W) X D
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine_per_expert = jax.nn.softmax(logits, axis=-1)

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for i in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    stack = []
                    y_tilda_tilda = jnp.einsum('bnpd,bmnp->bnmd', y_tilda, combine_per_expert)
                    # Output layer: one Q-value per action per expert
                    for i in range(self.config["NUM_EXPERTS"]):
                        # The input is a vector of shape M * D into the Q-value head
                        expert_perm_q_val = nn.Dense(self.action_dim)(y_tilda_tilda[:, i, :, :].reshape(B, -1))
                        stack.append(expert_perm_q_val)
                    y = jnp.stack(stack, axis=1) # BNA

                    if self.config["EXPERT_OUTPUT_COMBINE_STRAT"] == 'sum':
                        x = jnp.sum(y, axis=1)
                        return x
                    elif self.config["EXPERT_OUTPUT_COMBINE_STRAT"] == "softmax_over_n_meanpool":
                        combine_q_vectors = jax.nn.softmax(jnp.mean(logits, axis=(1, 3)), axis=1) # BN
                        x = jnp.einsum("bn,bna->ba", combine_q_vectors, y)
                        return x
                    else:
                        raise ValueError("Incorrect EXPERT_OUTPUT_COMBINE_STRAT")
                elif self.config['SOFT_MOE_APPR'] == 'big':
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
                        for i in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    x = jnp.einsum("bnpd,bmnp->bmd", y_tilda, combine)
                    x = nn.Dense(self.action_dim)(x.reshape(B, -1))
                    return x
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
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_pixels':
                x = x.astype(jnp.uint32) / 255.0
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'flattened':
                x = x.reshape(B, -1)
            elif self.config["FEATURES_FROM_PIXELS_STRAT"] == 'scaled_flattened':
                x = x.astype(jnp.uint32) / 255.0
                x = x.reshape(B, -1)
            else:
                raise ValueError("Wrong way to generate feature from pixels")


            if self.config['USE_SOFT_MOE_MULTI_EXPERT_TRANS']:

                assert 'flattened' not in self.config["FEATURES_FROM_PIXELS_STRAT"]

                if self.config["SOFT_MOE_APPR_TRANS"] == 'ours':
                    B, H, W, D = x.shape
                    print("Debug print x.shape after conv", x.shape)
                    #TOKENIZE PerConv
                    x = x.reshape(B, -1, D) # Shape (H*W) X D
                    # Let M = H * W
                    NUM_SLOTS_PER_EXPERT = (H * W) // self.config["NUM_EXPERTS"] # Each expert sort of gets equal number of tokens
                    phi = self.param("phi", nn.initializers.normal(), (D, self.config["NUM_EXPERTS"], NUM_SLOTS_PER_EXPERT)) # Shape: DNP
                    logits = jnp.einsum("bmd,dnp->bmnp", x, phi)

                    dispatch = jax.nn.softmax(logits, axis=1)
                    combine_per_expert = jax.nn.softmax(logits, axis=-1)

                    x_tilda = jnp.einsum("bmd,bmnp->bnpd", x, dispatch)

                    stack = []
                    for i in range(self.config["NUM_EXPERTS"]):
                        expert_out = x_tilda[:, i, :, :]
                        for i in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    stack = []
                    y_tilda_tilda = jnp.einsum('bnpd,bmnp->bnmd', y_tilda, combine_per_expert)
                    # Output layer: one Q-value per action per expert
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
                        for i in range(self.num_layers):
                            expert_out = nn.Dense(D)(expert_out)
                            expert_out = normalize(expert_out)
                            expert_out = nn.relu(expert_out)
                        stack.append(expert_out)
                    y_tilda = jnp.stack(stack, axis=1) # Shape: BNPD

                    x = jnp.einsum("bnpd,bmnp->bmd", y_tilda, combine)
                    x = nn.Dense(self.action_dim)(x.reshape(B, -1))
                    return x
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

        x = nn.Dense(self.action_dim)(x)

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
                lr = lr_scheduler if config.get("LR_PERM_LINEAR_DECAY", False) else config["LR"]
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
        else:
            train_state_perm = None

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, train_state_perm, expl_state, test_metrics, rng = runner_state

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
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
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
                    return (train_state, rng), (loss, qvals)

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
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
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
                            return loss, updates_perm

                        (loss_perm, updates_perm), grads = jax.value_and_grad(
                            _loss_fn_perm, has_aux=True
                        )(train_state_perm.params, train_state.params)
                        train_state_perm = train_state_perm.apply_gradients(grads=grads)
                        train_state_perm = train_state_perm.replace(
                            grad_steps=train_state_perm.grad_steps + 1,
                            batch_stats=updates_perm["batch_stats"],
                        )
                        return (train_state_perm, rng), loss_perm

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
                    (train_state_perm, rng), loss = jax.lax.scan(
                        _learn_phase_perm, (train_state_perm, rng), (minibatches, targets)
                    )

                    # Soft-reset transient. Keeping the transient-weight the same across the minibatches so only resetting at end of epoch.
                    train_state = train_state.replace(
                        params=jax.tree_map(
                            lambda x: (config["TRANSIENT_WEIGHT_DECAY"] ** (train_state_perm.n_updates)) *  x, train_state.params
                        ))

                    return (train_state_perm, train_state, rng), loss

                rng, _rng = jax.random.split(rng)
                is_perm_learn_time = (
                    train_state.n_updates % config["PERM_UPDATE_FREQ"] == 0
                )
                dummy_loss = jnp.zeros(
                    (config["NUM_EPOCHS"], config["NUM_MINIBATCHES"]))
                (train_state_perm, train_state, rng), loss_perm = jax.lax.cond(
                    is_perm_learn_time,
                    lambda train_state_perm, train_state, rng: jax.lax.scan(
                        _learn_epoch_perm, (train_state_perm, train_state,
                                            rng), None, config["NUM_EPOCHS"]
                    ),
                    lambda train_state_perm, train_state, rng: ((train_state_perm, train_state, rng), dummy_loss),
                    train_state_perm,
                    train_state,
                    _rng
                )


            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    
                    # log at intervals 
                    if (
                        metrics["update_steps"] % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
                        us = metrics["update_steps"]
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            old_env_steps = metrics["env_step"]
                            metrics = {
                                    f"rng{int(original_rng)}/{k}": v
                                    for k, v in metrics.items()
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
    )

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
