import numpy as np
import torch as th

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from functools import partial
from type_aliases import RNNStates
from ssd_methods import gather_nd, kl_div

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

import MOALayers as layers


class Soc_Inf_Policy(ActorCriticPolicy):
  """
  Policy class based on sb3s ActorCriticPolicy, that enables
  training agents using a causal influence
  reward, based on the description of Natasha Jaques et al. in the Paper
  Social Influence as IntrinsicMotivation for Multi-Agent Deep Reinforcement
  Learning (see https://arxiv.org/abs/1810.08647)

  :param observation_space: Observation space
  :param action_space: Action space
  :param lr_schedule: Learning rate schedule (could be constant)
  :param net_arch: The specification of the policy and value networks.
  :param activation_fn: Activation function
  :param ortho_init: Whether to use or not orthogonal initialization
  :param use_sde: Whether to use State Dependent Exploration or not
  :param log_std_init: Initial value for the log standard deviation
  :param full_std: Whether to use (n_features x n_actions) parameters
      for the std instead of only (n_features,) when using gSDE
  :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
      a positive standard deviation (cf paper). It allows to keep variance
      above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
  :param squash_output: Whether to squash the output using a tanh function,
      this allows to ensure boundaries when using gSDE.
  :param features_extractor_class: Features extractor to use.
  :param features_extractor_kwargs: Keyword arguments
      to pass to the features extractor.
  :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
  :param normalize_images: Whether to normalize images or not,
        dividing by 255.0 (True by default)
  :param optimizer_class: The optimizer to use,
      ``th.optim.Adam`` by default
  :param optimizer_kwargs: Additional keyword arguments,
      excluding the learning rate, to pass to the optimizer
  """

  def __init__(
      self,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      lr_schedule: Schedule,
      net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
      activation_fn: Type[nn.Module] = nn.Tanh,
      ortho_init: bool = False,
      use_sde: bool = False,
      log_std_init: float = 0.0,
      full_std: bool = True,
      use_expln: bool = False,
      squash_output: bool = False,
      features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
      features_extractor_kwargs: Optional[Dict[str, Any]] = None,
      normalize_images: bool = True,
      optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
      optimizer_kwargs: Optional[Dict[str, Any]] = None,
      div_measure="kl",
      fc_activation_fn: Type[nn.Module] = nn.Tanh,
      num_agents: int = 0,
      mixed: bool = False,
      # standardvalues taken from ssd-games
      num_frames=4,
      cell_size=128,
  ):
    assert num_agents > 0, "num_agents hasn't been initialized"

    self.num_agents = num_agents
    self.inf_threshold_reached = False
    self.activation_fn = fc_activation_fn
    self.div_measure = div_measure
    self.num_frames = num_frames
    self.cell_size = cell_size
    self.prev_features = None
    share_features_extractor = True
    self.use_inf_rew = np.zeros((num_agents))
    if mixed:
      for agent in range(int(num_agents / 2)):
        self.use_inf_rew[agent] = True

    super().__init__(
        observation_space,
        action_space,
        lr_schedule,
        net_arch,
        activation_fn,
        ortho_init,
        use_sde,
        log_std_init,
        full_std,
        use_expln,
        squash_output,
        features_extractor_class,
        features_extractor_kwargs,
        share_features_extractor,
        normalize_images,
        optimizer_class,
        optimizer_kwargs,
    )

  def _build_ac_network(self):
    self.ac_network = layers.AC_Net(
        # this only works like this because the action space for
        # each agent is discrete
        self.features_dim * 2,
        self.cell_size,
        action_out_size=self.action_space.n,
    )

  def _build(self, lr_schedule: Schedule) -> None:
    """
    Create the networks and the optimizer.

    :param lr_schedule: Learning rate schedule
        lr_schedule(1) is the initial learning rate
    """

    self._build_ac_network()

    latent_dim_pi = self.ac_network.get_act_out_size()

    if isinstance(self.action_dist, DiagGaussianDistribution):
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
          latent_dim=latent_dim_pi, log_std_init=self.log_std_init
      )
    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
          latent_dim=latent_dim_pi,
          latent_sde_dim=latent_dim_pi,
          log_std_init=self.log_std_init,
      )
    elif isinstance(
        self.action_dist,
        (
            CategoricalDistribution,
            MultiCategoricalDistribution,
            BernoulliDistribution,
        ),
    ):
      self.action_net = self.action_dist.proba_distribution_net(
          latent_dim=latent_dim_pi
      )
    else:
      raise NotImplementedError(
          f"Unsupported distribution '{self.action_dist}'."
      )

    # Init weights: use orthogonal initialization
    # with small initial weight for the output
    if self.ortho_init:
      # Values from stable-baselines.
      # features_extractor/mlp values are
      # originally from openai/baselines (default gains/init_scales).
      module_gains = {
          self.features_extractor: np.sqrt(2),
          self.mlp_extractor: np.sqrt(2),
          self.action_net: 0.01,
          self.value_net: 1,
      }
      if not self.share_features_extractor:
        # Note(antonin): this is to keep SB3 results
        # consistent, see GH#1148
        del module_gains[self.features_extractor]
        module_gains[self.pi_features_extractor] = np.sqrt(2)
        module_gains[self.vf_features_extractor] = np.sqrt(2)

      for module, gain in module_gains.items():
        module.apply(partial(self.init_weights, gain=gain))

    # Setup optimizer with initial learning rate
    self.optimizer = self.optimizer_class(
        self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
    )

  def get_counterfacts(
      self, obs, agent_idx, other_agents_acts, lstm_states, episode_starts
  ):

    counterfactual_preds = []

    # we know that meltingpot only uses discrete action spaces, so we can use
    # action_space.n to get the number of possible actions
    for i in range(self.action_space.n):
      actions_with_counterfactual = th.cat([
          other_agents_acts[:agent_idx],
          th.Tensor([i]),
          other_agents_acts[agent_idx:],
      ])
      repeated_acts = actions_with_counterfactual.repeat(
          self.observation_space.shape
      ).reshape((-1,) + self.observation_space.shape)
      observations = th.cat([obs, repeated_acts])

      features_origin = self.extract_features(observations)
      features = features_origin.reshape(self.num_agents, -1)

      counterfact_pred, _, _ = self.ac_network.forward(
          features, lstm_states, episode_starts
      )
      counterfact_pred = th.cat(
          [counterfact_pred[:agent_idx], counterfact_pred[agent_idx + 1 :]]
      )
      counterfactual_preds.append(counterfact_pred)

    counterfactuals = th.cat(counterfactual_preds, dim=-2)
    counterfactuals = counterfactuals.reshape(
        -1, counterfactuals.shape[-1], counterfactuals.shape[-2]
    )
    return counterfactuals

  def forward(
      self,
      obs: th.Tensor,
      lstm_states: Tuple,
      episode_starts: th.Tensor,
      prev_acts: th.Tensor,
      deterministic=False,
  ):
    """
    Forward pass in all networks (actor and critic)

    Args:
        obs (th.Tensor): the current observations made by agents in the environment
        lstm_states (RNNStates): the current lstm_states for the ac_lstm
        episode_starts (th.Tensor): current episode starts needed for the lstm
        prev_acts (th.Tensor): the acts from the last step, needed to feed
                            into the ac_network, as those acts should
                            influence the agents in their behavior so the
                            influence reward can have some effect.
        deterministic (bool, optional): Indicator if actions should
                                        be generated deterministically or not.
                                        Defaults to False.
    """
    if isinstance(prev_acts, np.ndarray):
      prev_acts = th.from_numpy(prev_acts).flatten()
    repeated_acts = prev_acts.repeat(self.observation_space.shape).reshape(
        (-1,) + self.observation_space.shape
    )
    observations = th.cat((obs, repeated_acts))

    features = self.extract_features(observations)
    features = features.reshape(self.num_agents, -1)

    latent_pi, values, new_ac_lstm_states = self.ac_network.forward(
        features, lstm_states, episode_starts
    )
    distribution = self._get_action_dist_from_latent(latent_pi)
    actions = distribution.get_actions(deterministic=deterministic)
    log_prob = distribution.log_prob(actions)
    actions = actions.reshape((-1, *self.action_space.shape))

    inf_rew = np.zeros((self.num_agents))
    if self.inf_threshold_reached:
      for agent in range(self.num_agents):
        if self.use_inf_rew[agent]:
          other_agent_acts = th.cat([prev_acts[:agent], prev_acts[agent + 1 :]])

          counterfacts = self.get_counterfacts(
              obs, agent, other_agent_acts, lstm_states, episode_starts
          )
          others_latent_pi = th.cat([latent_pi[:agent], latent_pi[agent + 1 :]])

          inf_rew[agent] = self.calc_influence_reward(
              prev_acts[agent], others_latent_pi, counterfacts
          )
    return actions, values, log_prob, new_ac_lstm_states, inf_rew

  def eval_forward(
      self,
      obs: th.Tensor,
      lstm_states: Tuple,
      episode_starts: th.Tensor,
      prev_acts: th.Tensor,
      deterministic=False,
  ):
    """
    shortened forward for evaluation

    Args:
        obs (th.Tensor): the current observations made by agents in the environment
        lstm_states (RNNStates): the current lstm_states for the ac_lstm
        episode_starts (th.Tensor): current episode starts needed for the lstm
        prev_acts (th.Tensor): the acts from the last step, needed to feed
                            into the ac_network, as those acts should
                            influence the agents in their behavior so the
                            influence reward can have some effect.
        deterministic (bool, optional): Indicator if actions should
                                        be generated deterministically or not.
                                        Defaults to False.
    """
    self.set_training_mode(False)

    if isinstance(prev_acts, np.ndarray):
      prev_acts = th.from_numpy(prev_acts).flatten()
    repeated_acts = prev_acts.repeat(self.observation_space.shape).reshape(
        (-1,) + self.observation_space.shape
    )
    observations = th.cat((obs, repeated_acts))

    features = self.extract_features(observations)
    features = features.reshape(self.num_agents, -1)

    latent_pi, _, new_ac_lstm_states = self.ac_network.forward(
        features, lstm_states, episode_starts
    )
    distribution = self._get_action_dist_from_latent(latent_pi)
    actions = distribution.get_actions(deterministic=deterministic)
    actions = actions.reshape((-1, *self.action_space.shape))

    inf_rew = np.zeros((self.num_agents))
    for agent in range(self.num_agents):
      other_agent_acts = th.cat([prev_acts[:agent], prev_acts[agent + 1 :]])

      counterfacts = self.get_counterfacts(
          obs, agent, other_agent_acts, lstm_states, episode_starts
      )
      others_latent_pi = th.cat([latent_pi[:agent], latent_pi[agent + 1 :]])

      inf_rew[agent] = self.calc_influence_reward(
          prev_acts[agent], others_latent_pi, counterfacts
      )
    return actions, new_ac_lstm_states, inf_rew

  def calc_influence_reward(
      self, agents_act_tensor, action_logits, counterfactual_logits
  ):
    """
    Compute influence of this agent on other agents.
    :param prev_action_logits: Logits for the agent's own policy/actions at t-1
    :prev_acts: the previous actions of all other agents
    :param counterfactual_logits: The counterfactual action logits for actions made by other
    agents at t.
    """
    # prev actions sollte basierend auf den Kommentaren in ssd die Aktionen des letzten
    # steps enthalten
    agent_actions_pre_last = th.reshape(agents_act_tensor, [-1, 1, 1]).type(
        th.int32
    )
    softmax = th.nn.Softmax()

    predicted_logits = gather_nd(
        params=counterfactual_logits, indices=agent_actions_pre_last
    )

    predicted_logits = th.reshape(
        predicted_logits, [-1, self.num_agents - 1, self.action_space.n]
    )
    predicted_logits = softmax(predicted_logits)
    predicted_logits = predicted_logits / th.sum(
        predicted_logits, dim=-1, keepdim=True
    )
    marginal_logits = self.marginalize_predictions(
        action_logits, counterfactual_logits
    )

    if self.div_measure == "kl":
      influence_reward = kl_div(predicted_logits, marginal_logits)
    elif self.div_measure == "jsd":
      mean_probs = 0.5 * (predicted_logits + marginal_logits)
      influence_reward = 0.5 * kl_div(
          predicted_logits, mean_probs
      ) + 0.5 * kl_div(marginal_logits, mean_probs)
    else:
      print("influence measure is not implemented, using kl as default...")
      influence_reward = kl_div(predicted_logits, marginal_logits)

    influence_reward = th.sum(influence_reward)
    return influence_reward

  def marginalize_predictions(self, prev_action_logits, counterfactual_logits):
    """
    Calculates marginal policies for all other agents.
    :param prev_action_logits: The agent's own policy logits at time t-1 .
    :param counterfactual_logits: The counterfactual action predictions made at time t-1 for
    other agents' actions at t.
    :return: The marginal policies for all other agents.
    """
    # normalize probabilities of original actions
    softmax = th.nn.Softmax()
    logits = softmax(prev_action_logits)
    logits = logits / th.sum(logits, dim=-1, keepdim=True)

    # change indexing from [B, num_actions, num_other_agents * num_actions]
    # to [B, num_actions, num_other_agents, num_actions]
    counterfactual_logits = th.reshape(
        counterfactual_logits,
        [-1, self.action_space.n, self.num_agents - 1, self.action_space.n],
    )
    counterfactual_logits = softmax(counterfactual_logits)

    # Change shape to broadcast probability of each action over counterfactual actions
    logits = th.reshape(logits, [-1, self.action_space.n, 1, 1])

    normalized_counterfacts = logits * counterfactual_logits
    # Remove counterfactual action dimension
    marginal_probs = th.sum(normalized_counterfacts, dim=-3)
    # normalize
    marginal_probs = marginal_probs / th.sum(
        marginal_probs, dim=-1, keepdim=True
    )

    return marginal_probs

  def inf_threshold_is_reached(self):
    self.inf_threshold_reached = True

  def predict_values(
      self,
      obs: th.Tensor,
      lstm_states: Tuple[th.Tensor, th.Tensor],
      episode_starts: th.Tensor,
      prev_acts: th.Tensor,
  ) -> th.Tensor:
    """
    Get the estimated values according to the current policy given the observations.

    :param obs: Observation.
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :param prev_acts: the actions from the previous step, needed to be fed into
                      the ac_network
    :return: the estimated values.
    """
    if isinstance(episode_starts, np.ndarray):
      episode_starts = th.from_numpy(episode_starts)
    if isinstance(prev_acts, np.ndarray):
      prev_acts = th.from_numpy(prev_acts).flatten()

    repeated_acts = prev_acts.repeat(self.observation_space.shape).reshape(
        (-1,) + self.observation_space.shape
    )
    obs = th.cat((obs, repeated_acts))

    features = self.extract_features(obs)
    features = features.reshape(int(features.shape[0] / 2), -1)

    _, value, _ = self.ac_network.forward(features, lstm_states, episode_starts)

    return value

  def evaluate_actions(
      self, obs, actions, prev_acts: th.Tensor, lstm_states, episode_starts
  ):
    """
    Evaluate actions according to the current policy,
    given the observations.

    :param obs: Observation.
    :param actions: all actions (only consider those of others)
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :return: estimated value, log likelihood of taking those actions
        and entropy of the action distribution.
    """
    if isinstance(episode_starts, np.ndarray):
      episode_starts = th.from_numpy(episode_starts)
    if isinstance(prev_acts, np.ndarray):
      prev_acts = th.from_numpy(prev_acts).flatten()

    repeated_acts = prev_acts.repeat(self.observation_space.shape).reshape(
        (-1,) + self.observation_space.shape
    )
    obs = th.cat((obs, repeated_acts))

    features = self.extract_features(obs)
    features = features.reshape(int(features.shape[0] / 2), -1)

    latent_pi, values, _ = self.ac_network.forward(
        features, lstm_states, episode_starts
    )

    distribution = self._get_action_dist_from_latent(latent_pi)

    log_prob = distribution.log_prob(actions)

    return values, log_prob, distribution.entropy()

  def predict(
      self, observation, prev_actions, state, episode_start, deterministic=False
  ) -> Tuple[np.ndarray | Tuple[np.ndarray] | None]:
    """
    Get the policy action from an observation (and optional hidden state).
    Includes sugar-coating to handle different observations (e.g. normalizing images).
    mostly copied from sb3s BasePolicy, but adapted to ac_Network

    :param observation: the input observation
    :param state: The last hidden states (can be None, used in recurrent policies)
    :param episode_start: The last masks (can be None, used in recurrent policies)
      this correspond to beginning of episodes,
      where the hidden states of the RNN must be reset.
    :param deterministic: Whether or not to return deterministic actions.
    :return: the model's action and the next hidden state
      (used in recurrent policies)
    """
    # Switch to eval mode (this affects batch norm / dropout)
    self.set_training_mode(False)

    # Check for common mistake that the user does not mix Gym/VecEnv API
    # Tuple obs are not supported by SB3, so we can safely do that check
    if (
        isinstance(observation, tuple)
        and len(observation) == 2
        and isinstance(observation[1], dict)
    ):
      raise ValueError(
          "You have passed a tuple to the predict() function instead of a Numpy"
          " array or a Dict. "
          "You are probably mixing Gym API with SB3 VecEnv API: `obs, info ="
          " env.reset()` (Gym) "
          "vs `obs = vec_env.reset()` (SB3 VecEnv). "
          "See related issue"
          " https://github.com/DLR-RM/stable-baselines3/issues/1694 "
          "and documentation for more information:"
          " https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
      )

    obs_tensor, vectorized_env = self.obs_to_tensor(observation)

    with th.no_grad():
      actions, state = self._predict(
          obs_tensor,
          prev_actions,
          state,
          episode_start,
          deterministic=deterministic,
      )
    # Convert to numpy, and reshape to the original action shape
    actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

    if isinstance(self.action_space, spaces.Box):
      if self.squash_output:
        # Rescale to proper domain when using squashing
        actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
      else:
        # Actions could be on arbitrary scale, so clip the actions to avoid
        # out of bound error (e.g. if sampling from a Gaussian distribution)
        actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

    # Remove batch dimension if needed
    if not vectorized_env:
      assert isinstance(actions, np.ndarray)
      actions = actions.squeeze(axis=0)

    return actions, state

  def _predict(
      self,
      observation,
      prev_actions,
      state,
      episode_start,
      deterministic: bool = False,
  ) -> th.Tensor:
    """
    Get the action according to the policy for a given observation.

    :param observation:
    :param deterministic: Whether to use stochastic or deterministic actions
    :return: Taken action according to the policy
    """
    dist, state = self.get_distribution(
        observation, prev_actions, state, episode_start
    )
    return dist.get_actions(deterministic=deterministic), state

  def get_distribution(
      self, obs, prev_acts, lstm_states, episode_starts
  ) -> Distribution:
    if isinstance(prev_acts, np.ndarray):
      prev_acts = th.from_numpy(prev_acts).flatten()
    repeated_acts = prev_acts.repeat(self.observation_space.shape).reshape(
        (-1,) + self.observation_space.shape
    )
    observations = th.cat((obs, repeated_acts))

    features = self.extract_features(observations)
    features = features.reshape(self.num_agents, -1)
    latent_pi, _, state = self.ac_network.forward(
        features, lstm_states, episode_starts
    )
    return self._get_action_dist_from_latent(latent_pi), state
