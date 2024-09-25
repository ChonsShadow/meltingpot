import warnings
import math
from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from Soc_Inf_policy import Soc_Inf_Policy
from buffers import Soc_Inf_Buffer

class Soc_Inf_ppo(PPO):
  """
  A Variant of sb3s PPO that is able to train agents using a causal influence
  reward, based on the description of Natasha Jaques et al. in the Paper
  Social Influence as IntrinsicMotivation for Multi-Agent Deep Reinforcement
  Learning (see https://arxiv.org/abs/1810.08647)

  Args:
      PPO (_type_): sb3s PPO

  :param policy: The policy model to use (should be soc_inf_policy Object, as
                 that policy is tailored to calculate and utilize social
                 Influence rewards
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
  def __init__(
      self,
      policy: Type[Soc_Inf_Policy],
      env: Union[GymEnv, str],
      num_agents: int,
      learning_rate: Union[float, Schedule] = 3e-4,
      inf_threshold: float = math.inf,
      n_steps: int = 2048,
      batch_size: int = 64,
      n_epochs: int = 10,
      gamma: float = 0.99,
      gae_lambda: float = 0.95,
      clip_range: Union[float, Schedule] = 0.2,
      clip_range_vf: Union[None, float, Schedule] = None,
      normalize_advantage: bool = True,
      ent_coef: float = 0.0,
      vf_coef: float = 0.5,
      max_grad_norm: float = 0.5,
      use_sde: bool = False,
      sde_sample_freq: int = -1,
      rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
      rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
      target_kl: Optional[float] = None,
      stats_window_size: int = 1000,
      tensorboard_log: Optional[str] = None,
      policy_kwargs: Optional[Dict[str, Any]] = None,
      verbose: int = 0,
      seed: Optional[int] = None,
      device: Union[th.device, str] = "auto",
      _init_setup_model: bool = True,
  ):
    super().__init__(
        policy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        sde_sample_freq=sde_sample_freq,
        rollout_buffer_class=rollout_buffer_class,
        rollout_buffer_kwargs=rollout_buffer_kwargs,
        stats_window_size=stats_window_size * num_agents,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device=device,
        seed=seed,
        _init_setup_model=False,
    )
    self.batch_size = int(self.n_envs * n_steps / 100)
    self.n_epochs = n_epochs
    self.clip_range = clip_range
    self.clip_range_vf = clip_range_vf
    self.normalize_advantage = normalize_advantage
    self.target_kl = target_kl
    self._last_lstm_states = None
    self.num_agents = num_agents
    self.agent_lables = []
    # noop for all, as nothing has happened yet
    self.prev_acts = th.zeros(num_agents)
    self.prev_episode_starts = np.zeros(num_agents)
    self.rew_instances = 0
    self.inf_threshold = inf_threshold * self.num_agents

    if _init_setup_model:
      self._setup_model()

  def _setup_model(self):
    self._setup_lr_schedule()
    self.set_random_seed(self.seed)

    buffer_cls = Soc_Inf_Buffer

    self.policy = self.policy_class(
        self.observation_space,
        self.action_space,
        self.lr_schedule,
        self.num_agents,
        use_sde=self.use_sde,
        **self.policy_kwargs,
    )
    self.policy.to(self.device)

    # Actor and critic use the same lstm, thus we only need
    # that one
    lstm = self.policy.ac_network.lstm
    single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
    # hidden and cell states for actor and critic
    self._last_lstm_states = (
        th.zeros(single_hidden_state_shape, device=self.device),
        th.zeros(single_hidden_state_shape, device=self.device),
    )

    hidden_state_buffer_shape = (
        self.n_steps,
        lstm.num_layers,
        self.n_envs,
        lstm.hidden_size,
    )

    self.rollout_buffer = buffer_cls(
        self.n_steps,
        self.observation_space,
        self.action_space,
        self.num_agents,
        hidden_state_buffer_shape,
        self.device,
        gamma=self.gamma,
        gae_lambda=self.gae_lambda,
        n_envs=self.n_envs,
    )

    # Initialize schedules for policy/value clipping
    self.clip_range = get_schedule_fn(self.clip_range)
    if self.clip_range_vf is not None:
      if isinstance(self.clip_range_vf, (float, int)):
        assert self.clip_range_vf > 0, (
            "`clip_range_vf` must be positive, pass `None` to deactivate vf"
            " clipping"
        )

        self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

  def collect_rollouts(
      self,
      env: VecEnv,
      callback: BaseCallback,
      rollout_buffer: RolloutBuffer,
      n_rollout_steps: int,
  ) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param rollout_buffer: Buffer to fill with rollouts
    :param n_steps: Number of experiences to collect per environment
    :return: True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    """
    assert self._last_obs is not None, "No previous observation was provided"
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)
    self.pred_actions = None

    n_steps = 0
    rollout_buffer.reset()

    callback.on_rollout_start()

    lstm_states = deepcopy(self._last_lstm_states)

    while n_steps < n_rollout_steps:
      if (
          self.use_sde
          and self.sde_sample_freq > 0
          and n_steps % self.sde_sample_freq == 0
      ):
        # Sample a new noise matrix
        self.policy.reset_noise(env.num_envs)

      with th.no_grad():
        # Convert to pytorch tensor or to TensorDict
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        episode_starts = th.tensor(
            self._last_episode_starts,
            dtype=th.float32,
            device=self.device,
        )
        actions, values, log_probs, lstm_states, inf_rews = self.policy(
            obs_tensor, lstm_states, episode_starts, self.prev_acts
        )
      actions = actions.cpu().numpy()

      # Rescale and perform action
      clipped_actions = actions

      if isinstance(self.action_space, spaces.Box):
        if self.policy.squash_output:
          # Unscale the actions to match env bounds
          # if they were previously squashed (scaled in [-1, 1])
          clipped_actions = self.policy.unscale_action(clipped_actions)
        else:
          # Otherwise, clip the actions to avoid out of bound error
          # as we are sampling from an unbounded Gaussian distribution
          clipped_actions = np.clip(
              actions, self.action_space.low, self.action_space.high
          )

      new_obs, rewards, dones, infos = env.step(clipped_actions)
      self.num_timesteps += env.num_envs
      if not self.policy.inf_threshold_reached:
        num_rews = 0
        for rew in rewards:
          if rew > 0:
            num_rews += 1

        self.rew_instances += num_rews

        if self.rew_instances >= self.inf_threshold:
          self.policy.inf_threshold_is_reached()

      # Give access to local variables
      callback.update_locals(locals())
      if not callback.on_step():
        return False

      self.pred_actions = None

      # pure_rews = rewards.copy()

      if inf_rews.any():
        self.rollout_buffer.add_inf_rew(inf_rews)

      self._update_info_buffer(infos, dones)

      n_steps += 1

      if isinstance(self.action_space, spaces.Discrete):
        # Reshape in case of discrete action
        actions = actions.reshape(-1, 1)

      # Handle timeout by bootstraping with value function
      # see GitHub issue #633
      for idx, done in enumerate(dones):
        if (
            done
            and infos[idx].get("terminal_observation") is not None
            and infos[idx].get("TimeLimit.truncated", False)
        ):
          terminal_obs = self.policy.obs_to_tensor(
              infos[idx]["terminal_observation"]
          )[0]
          with th.no_grad():
            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
          rewards[idx] += self.gamma * terminal_value

      rollout_buffer.add(
          self._last_obs,
          actions,
          rewards,
          self._last_episode_starts,
          values,
          log_probs,
          lstm_states=lstm_states,
      )

      self.prev_acts = actions
      self._last_obs = new_obs
      self._last_episode_starts = dones
      self._last_lstm_states = lstm_states

    with th.no_grad():
      # Compute value for the last timestep
      values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states, self._last_episode_starts, self.prev_acts)  # type: ignore[arg-type]

    rollout_buffer.compute_returns_and_advantage(
        last_values=values, dones=dones
    )

    callback.update_locals(locals())

    callback.on_rollout_end()

    return True

  def train(self) -> None:
    """
    Update policy using the currently gathered rollout buffer.
    """

    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    continue_training = True
    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []

      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions

        if isinstance(self.action_space, spaces.Discrete):
          # Convert discrete action from float to long
          actions = rollout_data.actions.long().flatten()

          # convert mask from float to bool
          mask = rollout_data.mask > 1e-8

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
          self.policy.reset_noise(self.batch_size)

        values, log_prob, entropy = self.policy.evaluate_actions(
            rollout_data.observations,
            actions,
            rollout_data.prev_actions,
            rollout_data.lstm_states,
            rollout_data.episode_starts,
        )
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage:
          advantages = (advantages - advantages[mask].mean()) / (
              advantages[mask].std() + 1e-8
          )

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(
            ratio, 1 - clip_range, 1 + clip_range
        )
        policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the different between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = rollout_data.old_values + th.clamp(
              values - rollout_data.old_values, -clip_range_vf, clip_range_vf
          )
        # Value loss using the TD(gae_lambda) target
        # Mask padded sequences
        value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -th.mean(-log_prob[mask])
        else:
          entropy_loss = -th.mean(entropy[mask])

        entropy_losses.append(entropy_loss.item())

        loss = (
            policy_loss
            + self.ent_coef * entropy_loss
            + self.vf_coef * value_loss
        )

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
          log_ratio = log_prob - rollout_data.old_log_prob
          approx_kl_div = (
              th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
          )
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(
                f"Early stopping at step {epoch} due to reaching max kl:"
                f" {approx_kl_div:.2f}"
            )
            break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.policy.optimizer.step()

      if not continue_training:
        break

    self._n_updates += self.n_epochs
    explained_var = explained_variance(
        self.rollout_buffer.values.flatten(),
        self.rollout_buffer.returns.flatten(),
    )

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record(
        "train/n_updates", self._n_updates, exclude="tensorboard"
    )
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)

  def learn(
      self,
      total_timesteps: int,
      callback: MaybeCallback = None,
      log_interval: int = 1,
      tb_log_name: str = "SocInfPPO",
      reset_num_timesteps: bool = True,
      progress_bar: bool = False,
  ):
    return super().learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=log_interval,
        tb_log_name=tb_log_name,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=progress_bar,
    )

  def predict(
      self,
      observation: Union[np.ndarray, Dict[str, np.ndarray]],
      state: Optional[Tuple[np.ndarray, ...]] = None,
      episode_start: Optional[np.ndarray] = None,
      deterministic: bool = False,
  ):
    if self.pred_actions is None:
      self.pred_actions = th.zeros(self.num_agents)
    if state == None:
      state = (
          th.zeros(self._last_lstm_states[0].shape, device=self.device),
          th.zeros(self._last_lstm_states[0].shape, device=self.device),
      )
    if isinstance(episode_start, np.ndarray):
      episode_start = th.from_numpy(episode_start).type(th.float32)
    self.pred_actions, state = self.policy.predict(
        observation, self.pred_actions, state, episode_start, deterministic
    )

    return self.pred_actions, state
