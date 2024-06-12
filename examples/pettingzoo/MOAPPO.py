from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from MOAPolicy import MOAPolicy
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from type_aliases import RNNStates

class MOAPPO(OnPolicyAlgorithm):
  """
      Proximal Policy Optimization algorithm (PPO) (clip version)
      with support for recurrent policies and decentralized learning.

      Based on the original Stable Baselines 3 implementation and the recurrent
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

        for i in range(num_agents):
            self.agent_lables.append("agent-" + str(i))

        if _init_setup_model:
            self._setup_model()

  def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RecurrentRolloutBuffer

        agents_policies = {}

        for agent in self.agent_lables:
            agents_policies[agent] = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs)
            agents_policies[agent].to(self.device)
            # the lstms have different purposes, thus they surely may have different architectures
            ac_lstm = agents_policies[agent].ac_lstm
            moa_lstm = agents_policies[agent].moa_lstm

            ac_single_hidden_state_shape = (ac_lstm.num_layers, self.n_envs, ac_lstm.hidden_size)
            moa_single_hidden_state_shape = (moa_lstm.num_layers, self.n_envs, moa_lstm.hidden_size)
            # states for ac und moa lstms:
            self._last_lstm_states = RNNStates(
                (
                    th.zeros(ac_single_hidden_state_shape, device=self.device),
                    th.zeros(ac_single_hidden_state_shape, device=self.device)
                ),
                (
                    th.zeros(moa_single_hidden_state_shape, device=self.device),
                    th.zeros(moa_single_hidden_state_shape, device=self.device)
                )
            )

            ac_hidden_state_buffer_shape = (self.n_steps, ac_lstm.num_layers,
                                            self.n_envs, ac_lstm.hidden_size)
            # TODO: include in new Bufferclass
            moa_hidden_state_buffer_shape = (self.n_steps, moa_lstm.num_layers,
                                             self.n_envs, moa_lstm.hidden_size)

            # TODO: create buffer class fitting the moa-Policy and if necessary,
            #       the decentralized learning approach
            self.rollout_buffer = buffer_cls(
              self.n_steps,
              self.observation_space,
              self.action_space,
              ac_hidden_state_buffer_shape,
              self.device,
              gamma=self.gamma,
              gae_lambda=self.gae_lambda,
              n_envs=self.n_envs,
            )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
          if isinstance(self.clip_range_vf, (float, int)):
              assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

          self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
