from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from MOAPolicy import MOAPolicy
from gymnasium import spaces
from typing import Tuple, List
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer, MOABuffer
from type_aliases import RNNStates


class MOAPPO(OnPolicyAlgorithm):
  """
  Proximal Policy Optimization algorithm (PPO) (clip version)
  with support for recurrent policies and decentralized learning.

  NOTE: Based on the original Stable Baselines 3 implementation and the recurrent
  implementation from sb3-contrib

  Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

  :param policy: The policy model to use (Default: MOAPolicy)
  :param env: The environment to learn from (if registered in Gym, can be str)
  :param num_agents: the number of agents; for each agent an individual policy
      will be created, to ensure decentralized learning
  :param learning_rate: The learning rate, it can be a function
      of the current progress remaining (from 1 to 0)
  :param n_steps: The number of steps to run for each environment per update
      (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
  :param batch_size: Minibatch size
  :param n_epochs: Number of epoch when optimizing the surrogate loss
  :param gamma: Discount factor
  :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
  :param clip_range: Clipping parameter, it can be a function of the current progressforw
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
  :param target_kl: Limit the KL divergence between updates,
      because the clipping is not enough to prevent large update
      see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
      By default, there is no limit on the kl div.
  :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
      the reported success rate, mean episode length, and mean reward over
  :param tensorboard_log: the log location for tensorboard (if None, no logging)
  :param policy_kwargs: additional arguments to be passed to the policy on creation
  :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
  :param seed: Seed for the pseudo random generators
  :param device: Device (cpu, cuda, ...) on which the code should be run.
      Setting it to auto, the code will be run on the GPU if possible.
  :param _init_setup_model: Whether or not to build the network at the creation of the instance
  """

  def __init__(
      self,
      policy: Type[MOAPolicy],
      env: Union[GymEnv, str],
      num_agents: int,
      learning_rate: Union[float, Schedule] = 3e-4,
      n_steps: int = 128,
      batch_size: Optional[int] = 128,
      n_epochs: int = 10,
      gamma: float = 0.99,
      gae_lambda: float = 0.95,
      clip_range: Union[float, Schedule] = 0.2,
      clip_range_vf: Union[None, float, Schedule] = None,
      normalize_advantage: bool = True,
      ent_coef: float = 0.0,
      vf_coef: float = 0.5,
      moa_coef: float = 1.0,
      max_grad_norm: float = 0.5,
      use_sde: bool = False,
      sde_sample_freq: int = -1,
      target_kl: Optional[float] = None,
      stats_window_size: int = 100,
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
        stats_window_size=stats_window_size,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
        _init_setup_model=False,
        supported_action_spaces=(
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        ),
    )

    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.clip_range = clip_range
    self.clip_range_vf = clip_range_vf
    self.normalize_advantage = normalize_advantage
    self.target_kl = target_kl
    self._last_lstm_states = None
    self.num_agents = num_agents
    self.agent_lables = []
    self.moa_coef = moa_coef
    # noop for all, as nothing has happened yet
    self.prev_acts = th.zeros(num_agents)
    self.prev_episode_starts = np.zeros(num_agents)

    if _init_setup_model:
      self._setup_model()

  def _setup_model(self):
    self._setup_lr_schedule()
    self.set_random_seed(self.seed)

    buffer_cls = MOABuffer

    self.agents_policies = []
    self._last_lstm_states = []

    for agent in range(self.num_agents):
      self.agents_policies.append(
          self.policy_class(
              self.observation_space,
              self.action_space,
              self.lr_schedule,
              self.num_agents - 1,
              use_sde=self.use_sde,
              **self.policy_kwargs,
          )
      )
      self.agents_policies[agent].to(self.device)
      # the lstms have different purposes, thus they surely may have different architectures
      ac_lstm = self.agents_policies[agent].ac_lstm.lstm
      moa_lstm = self.agents_policies[agent].moa_lstm.lstm

      ac_single_hidden_state_shape = (
          ac_lstm.num_layers,
          int(self.n_envs / self.num_agents),
          ac_lstm.hidden_size,
      )
      moa_single_hidden_state_shape = (
          moa_lstm.num_layers,
          int(self.n_envs / self.num_agents),
          moa_lstm.hidden_size,
      )
      # states for ac und moa lstms:
      self._last_lstm_states.append(
          RNNStates(
              (
                  th.zeros(ac_single_hidden_state_shape, device=self.device),
                  th.zeros(ac_single_hidden_state_shape, device=self.device),
              ),
              (
                  th.zeros(moa_single_hidden_state_shape, device=self.device),
                  th.zeros(moa_single_hidden_state_shape, device=self.device),
              ),
          )
      )

      ac_hidden_state_buffer_shape = (
          self.n_steps,
          self.num_agents,
          ac_lstm.num_layers,
          int(self.n_envs / self.num_agents),
          ac_lstm.hidden_size,
      )

      moa_hidden_state_buffer_shape = (
          self.n_steps,
          self.num_agents,
          moa_lstm.num_layers,
          int(self.n_envs / self.num_agents),
          moa_lstm.hidden_size,
      )

      self.rollout_buffer = buffer_cls(
          self.n_steps,
          self.observation_space,
          self.action_space,
          ac_hidden_state_buffer_shape,
          moa_hidden_state_buffer_shape,
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
    # TODO: handle multiple envs!
    # Switch to eval mode (this affects batch norm / dropout)
    for agent in range(self.num_agents):
      self.agents_policies[agent].set_training_mode(False)
      # Sample new weights for the state dependent exploration
      if self.use_sde:
        self.agents_policies.reset_noise(env.num_envs)

    n_steps = 0
    rollout_buffer.reset()

    callback.on_rollout_start()

    lstm_states = deepcopy(self._last_lstm_states)

    while n_steps < n_rollout_steps:
      agent_actions = []
      agent_values = []
      agent_log_probs = []
      clipped_actions = []
      agents_pred_acts = []
      inf_rews = np.zeros(self.num_agents)

      for agent in range(self.num_agents):
        if (
            self.use_sde
            and self.sde_sample_freq > 0
            and n_steps % self.sde_sample_freq == 0
        ):
          # Sample a new noise matrix
          self.agents_policies[agent].reset_noise(env.num_envs)

        with th.no_grad():
          # Convert to pytorch tensor or to TensorDict
          obs_tensor = th.as_tensor(
              self._last_obs[agent], dtype=th.float32, device=self.device
          )
          episode_starts = th.tensor(
              self._last_episode_starts[agent],
              dtype=th.float32,
              device=self.device,
          )

          (
              new_actions,
              values,
              log_probs,
              pred_actions,
              new_lstm_states,
              inf_rew,
          ) = self.agents_policies[agent].forward(
              obs_tensor,
              lstm_states[agent],
              episode_starts,
              self.prev_acts,
              agent,
              th.tensor(float(self.prev_episode_starts[agent])),
          )

          new_actions = new_actions.cpu().numpy()
          agent_actions.append(new_actions)
          agent_values.append(values)
          agent_log_probs.append(log_probs)
          agents_pred_acts.append(th.squeeze(pred_actions))
          inf_rews[agent] = th.sum(inf_rew)
          lstm_states[agent] = new_lstm_states

        clipped_actions.append(agent_actions[agent])

        if isinstance(self.action_space, spaces.Box):
          clipped_actions[agent] = np.clip(
              agent_actions[agent],
              self.action_space.low,
              self.action_space.high,
          )

      agent_actions = np.concatenate(agent_actions).ravel()
      clipped_actions = np.concatenate(clipped_actions).ravel()
      agent_values = th.cat(agent_values, dim=0)
      agent_log_probs = th.stack(agent_log_probs, dim=0).flatten()

      new_obs, rewards, dones, infos = env.step(clipped_actions)

      self.num_timesteps += 1

      # TODO: inf_rew is calculated for past action -> we need to add it to the
      # rew from that past action!
      pure_rews = rewards.copy()

      if inf_rews.any():
        self.rollout_buffer.add_inf_rew(inf_rews)

      # Give access to local variables
      callback.update_locals(locals())
      if not callback.on_step():
        return False

      self._update_info_buffer(infos, dones)

      n_steps += 1

      if isinstance(self.action_space, spaces.Discrete):
        # Reshape in case of discrete action
        for agent in range(self.num_agents):
          agent_actions[agent] = agent_actions[agent].reshape(-1, 1)

      # NOTE: this seems very strange, gym should be able to handle multiple agents
      # though sb3 seems to only consider different envs but one single agent
      for idx, done_ in enumerate(dones):
        if (
            done_
            and infos[idx].get("terminal_observation") is not None
            and infos[idx].get("TimeLimit.truncated", False)
        ):
          terminal_obs = self.agents_policies[idx].obs_to_tensor(
              infos[idx]["terminal_observation"]
          )[0]
          with th.no_grad():
            terminal_lstm_states = (
                lstm_states[idx].ac[0][:, idx : idx + 1, :].contiguous(),
                lstm_states[idx].ac[0][:, idx : idx + 1, :].contiguous(),
            )
            episode_starts = th.tensor(
                [False], dtype=th.float32, device=self.device
            )
            terminal_value = self.agents_policies[idx].predict_values(
                terminal_obs, terminal_lstm_states, episode_starts
            )[0]
          rewards[idx] += self.gamma * terminal_value
          pure_rews[idx] += self.gamma * terminal_value

      rollout_buffer.add(
          self._last_obs,
          th.from_numpy(agent_actions),
          rewards,
          self._last_episode_starts,
          agent_values,
          agent_log_probs,
          lstm_states=self._last_lstm_states,
          num_agents=self.num_agents,
          pred_actions=agents_pred_acts,
          pure_rews=pure_rews,
      )

      self.prev_acts = th.from_numpy(agent_actions)
      # TODO: come up with better names...
      self.prev_episode_starts = self._last_episode_starts
      self._last_obs = new_obs
      self._last_episode_starts = dones
      self._last_lstm_states = lstm_states

    with th.no_grad():
      # compute value for the last timestep
      agent_values = []
      for agent in range(self.num_agents):
        episode_starts = th.tensor(
            dones[agent], dtype=th.float32, device=self.device
        )
        agent_values.append(
            self.agents_policies[agent].predict_values(
                th.as_tensor(
                    new_obs[agent], dtype=th.float32, device=self.device
                ),
                lstm_states[agent].ac,
                episode_starts,
            )
        )

    rollout_buffer.compute_returns_and_advantage(
        last_values=agent_values, dones=dones, num_agents=self.num_agents
    )

    callback.on_rollout_end()

    return True

  def train(self) -> None:
    """
    Update policy using the currently gathered rollout buffer.
    """
    for agent in range(self.num_agents):
      # Switch to train mode (this affects batch norm / dropout)
      self.agents_policies[agent].set_training_mode(True)
      # Update optimizer learning rate
      self._update_learning_rate(self.agents_policies[agent].optimizer)

    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)
    # NOTE: this does not directly relate to the way the actor-critic model is build
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

    # for logging
    entropy_losses = []
    pg_losses, value_losses = [], []
    moa_losses = []
    clip_fractions = []
    losses = []

    # NOTE: individual training is dependent on the individual kl_div
    #       training in general is dependent on the mean kl_div between all agents
    #       it is possible that the coefficients for the target_kl must be adjusted
    #       appropriately
    continue_training = True

    for epoch in range(self.n_epochs):
      mean_approx_kl_divs = []
      # Do a complete pass on the rollout buffer (decentralized - for each agent one indiviual pass!)

      loss = []
      current_approx_kl_divs = []

      for agent in range(self.num_agents):
        for rollout_data in self.rollout_buffer.get(agent, self.batch_size):
          actions = rollout_data.actions
          if isinstance(self.action_space, spaces.Discrete):
            # convert discrete actions from float to long
            actions = rollout_data.actions.long().flatten()

          # convert mask from float to bool
          mask = rollout_data.mask > 1e-8

          # Re-sample the noise matrix because the log_std has changed
          if self.use_sde:
            self.agents_policies[agent].reset_noise(self.batch_size)

          value, log_prob, entropy = self.agents_policies[
              agent
          ].evaluate_actions(
              rollout_data.observations,
              actions,
              rollout_data.lstm_states,
              rollout_data.episode_starts,
          )

          value = value.flatten()

          # NOTE: maybe we need to inspect the shape of advantages, as it only refers to one agents
          #       advantages
          advantages = rollout_data.advantages
          if self.normalize_advantage:
            advantages = (advantages - advantages[mask].mean()) / (
                advantages[mask].std() + 1e-8
            )

          ratio = th.exp(log_prob - rollout_data.old_log_prob)

          # clipped surrogate loss
          policy_loss_1 = advantages * ratio
          policy_loss_2 = advantages * th.clamp(
              ratio, 1 - clip_range, 1 + clip_range
          )
          policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

          pg_losses.append(policy_loss.item())
          clip_fraction = th.mean(
              (th.abs(ratio - 1) > clip_range).float()
          ).item()
          clip_fractions.append(clip_fraction)

          if self.clip_range_vf is None:
            # No clipping
            values_pred = value
          else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                value - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )

          # Value loss using the TD(gae_lambda) target
          # Mask padded sequences
          value_loss = th.mean(
              ((rollout_data.returns - values_pred) ** 2)[mask]
          )

          value_losses.append(value_loss.item())

          # Entropy loss favor exploration
          if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
          else:
            entropy_loss = -th.mean(entropy[mask])

          entropy_losses.append(entropy_loss.item())

          moa_loss, _ = self.agents_policies[agent].calc_moa_loss(
              rollout_data.pred_actions, rollout_data.others_acts
          )

          moa_losses.append(moa_loss.item())

          loss = (
              policy_loss
              + self.ent_coef * entropy_loss
              + self.vf_coef * value_loss
              + self.moa_coef * moa_loss
          )

          losses.append(loss)

          with th.no_grad():
            log_ratio = log_prob - rollout_data.old_log_prob
            approx_kl_div = (
                th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask])
                .cpu()
                .numpy()
            )
            current_approx_kl_divs.append(approx_kl_div)

          # check if target kl is reached for individual agent:
          if (
              self.target_kl is not None
              and approx_kl_div > 1.5 * self.target_kl
          ):
            # stop training for current agent
            break

          # Optimization:
          self.agents_policies[agent].optimizer.zero_grad()
          th.autograd.set_detect_anomaly(True)
          loss.backward()
          th.autograd.set_detect_anomaly(False)
          # Clip grad norm
          th.nn.utils.clip_grad_norm_(
              self.agents_policies[agent].parameters(), self.max_grad_norm
          )
          self.agents_policies[agent].optimizer.step()

        # calculate mean kl_div to decide if training should be finished for now
        with th.no_grad():
          cur_mean_approx_kl_div = np.mean(current_approx_kl_divs)
          # for logging
          mean_approx_kl_divs.append(cur_mean_approx_kl_div)

        # has mean kl_div reached target_kl?
        if (
            self.target_kl is not None
            and cur_mean_approx_kl_div > 1.5 * self.target_kl
        ):
          # stop training alltogether
          continue_training = False
          if self.verbose >= 1:
            print(
                f"Early stopping at step {epoch} due to reaching max kl:"
                f" {cur_mean_approx_kl_div:.2f}"
            )
          break

      if not continue_training:
        break

    self._n_updates += self.n_epochs
    explained_var = explained_variance(
        self.rollout_buffer.values.flatten(),
        self.rollout_buffer.returns.flatten(),
    )

    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(current_approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)

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
      tb_log_name: str = "MOAPPO",
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

    actions = np.zeros((self.num_agents, *self.action_space.shape))
    new_states = False
    if state == None:
      new_states = True
      state = []
    episode_start = th.from_numpy(episode_start).type(th.float32)
    for agent in range(self.num_agents):
      if new_states:
        state.append(
          (th.zeros(self._last_lstm_states[0].ac[0].shape),
          th.zeros(self._last_lstm_states[0].ac[0].shape),)
        )
      obs_tensor = th.as_tensor(
          observation[agent], dtype=th.float32, device=self.device
      )
      actions[agent], state[agent] = self.agents_policies[agent].predict(
          obs_tensor, state[agent], episode_start[agent], deterministic
      )

    return actions.astype(int), state

  def _excluded_save_params(self) -> List[str]:
    """
    Returns the names of the parameters that should be excluded from being
    saved by pickling. E.g. replay buffers are skipped by default
    as they take up a lot of space. PyTorch variables should be excluded
    with this so they can be stored with ``th.save``.

    :return: List of parameters that should be excluded from being saved with pickle.
    """
    return [
        "agents_policies",
        "_last_lstm_states",
        "device",
        "env",
        "replay_buffer",
        "rollout_buffer",
        "_vec_normalize_env",
        "_episode_storage",
        "_logger",
        "_custom_logger",
    ]

  def _get_torch_save_params(self) -> Tuple[th.List[str]]:
    state_dicts = []
    for agent in range(self.num_agents):
      setattr(self, f"policy_{agent}", self.agents_policies[agent])
      state_dicts.append(f"policy_{agent}")
      state_dicts.append(f"policy_{agent}.optimizer")
    return state_dicts, []
