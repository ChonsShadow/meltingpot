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


class MOAPolicy(ActorCriticPolicy):
  """
  Policy class created for performing an actor critic algorithm with social influence rewards

  :param observation_space: Observation space
  :param action_space: Action space
  :param lr_schedule: Learning rate schedule (could be constant)
  :param net_arch: The specification of the policy, value and moa networks
  :param fc_activation_fn: Activation function for the ac and moa fc-layers
  :param conv_activation_fn: Activation function for the shared conv-layer
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
      num_other_agents,
      influence_only_when_visible,
      div_measure="kl",
      net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
      fc_activation_fn: Type[nn.Module] = nn.Tanh,
      conv_activation_fn: Type[nn.Module] = nn.ReLU,
      ortho_init: bool = False,
      use_sde: bool = False,
      log_std_init: float = 0.0,
      full_std: bool = True,
      use_expln: bool = False,
      squash_output: bool = False,
      features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
      features_extractor_kwargs: Optional[Dict[str, Any]] = None,
      share_features_extractor: bool = True,
      normalize_images: bool = True,
      optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
      optimizer_kwargs: Optional[Dict[str, Any]] = None,
      cell_size=128,  # standardvalue taken from ssd-games
  ):
    self.conv_activation_fn = conv_activation_fn
    self.cell_size = cell_size
    self.num_other_agents = num_other_agents
    self.influence_only_when_visible = influence_only_when_visible
    self.num_actions = get_action_dim(action_space)
    self.prev_actions = []
    self.div_measure = div_measure
    self.influence_reward = None
    self.prev_act_logits = None

    super.__init__(
        observation_space,
        action_space,
        lr_schedule,
        net_arch=net_arch,
        activation_fn=fc_activation_fn,
        ortho_init=ortho_init,
        use_sde=use_sde,
        log_std_init=log_std_init,
        full_std=full_std,
        use_expln=use_expln,
        squash_output=squash_output,
        share_features_extractor=share_features_extractor,
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        normalize_images=normalize_images,
    )

  def _build_MOAMLP(self) -> None:
    """
    create the layers needed for ac and moa
    """
    # Note: If net_arch is None and some features extractor is used,
    #       net_arch here is an empty list and mlp_extractor does not
    #       really contain any layers (acts like an identity module).
    self.mlp_extractor = layers.MOAMlp(
        self.features_dim,
        self.net_arch,
        self.conv_activation_fn,
        self.activation_fn,
        self.activation_fn,
        self.device,
    )

  def _build_ac_lstm(self):
    self.ac_lstm = layers.ACLSTM(
        self.mlp_extractor.get_ac_out_dim(), self.cell_size
    )

  def _build_moa_lstm(self, in_size, num_outputs, num_actions):
    self.moa_lstm = layers.MOALSTM(
        in_size, num_outputs, num_actions, self.cell_size
    )

  def _build(self, lr_schedule: Schedule) -> None:
    """
    Create the networks and the optimizer.

    :param lr_schedule: Learning rate schedule
        lr_schedule(1) is the initial learning rate
    """
    self._build_MOAMLP()
    self._build_ac_lstm()

    latent_dim_pi = self.ac_lstm.get_act_out_size()

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

    self._build_moa_lstm(
        self.mlp_extractor.get_moa_out_dim,
        self.num_other_agents * self.num_actions,
        self.num_actions,
        self.cell_size,
    )

    # Init weights: use orthogonal initialization
    # with small initial weight for the output
    if self.ortho_init:
      # TODO: check for features_extractor
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
      self, features, other_agents_acts, lstm_states, episode_starts
  ):
    counterfactual_preds = []
    for i in range(self.num_actions):
      actions_with_counterfactual = nn.functional.pad(
          other_agents_acts, (0, 1), mode="constant", value=i
      )
      moa_features = th.cat([features, actions_with_counterfactual])
      counterfact_pred, _ = self.moa_lstm.forward(
          moa_features, lstm_states, episode_starts
      )
      counterfactual_preds.append(counterfact_pred)

    counterfactuals = th.cat(counterfactual_preds, dim=-2)
    return counterfactuals

  def forward(
      self,
      obs,
      own_acts,
      other_agents_acts,
      lstm_states: RNNStates,
      episode_starts,
      deterministic,
      prev_obs,
      prev_acts,
      prev_episode_starts,
      visibility,
  ):
    """
    Forward pass in all the networks (actor, critic and moa)

    :param obs: Observation. Observation
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :param deterministic: Whether to sample or use deterministic actions
    :return: action, value and log probability of the action
    """
    features = self.extract_features(obs)

    latent_ac, latent_moa = self.mlp_extractor.forward(features)
    latent_pi, value, new_ac_lstm_states = self.ac_lstm.forward(
        latent_ac, lstm_states.ac, episode_starts
    )

    actions = th.cat([other_agents_acts, own_acts])
    flattened_acts = th.flatten(actions)
    moa_features = th.cat([latent_moa, flattened_acts])
    action_pred, new_moa_lstm_states = self.moa_lstm.forward(
        moa_features, lstm_states.moa, episode_starts
    )

    counterfacts = self.get_counterfacts(
        prev_obs, prev_acts, lstm_states.moa, prev_episode_starts
    )

    counterfacts = th.reshape(
        counterfacts, [-1, counterfacts.shape(-2), counterfacts.shape(-1)]
    )

    self.calc_influence_reward(self.prev_act_logits, counterfacts, visibility)

    self.prev_actions = own_acts
    self.prev_act_logits = action_pred

    action_dist = self._get_action_dist_from_latent(latent_pi)
    actions = action_dist.get_actions(deterministic=deterministic)
    log_prob = action_dist.log_prob(actions)

    return (
        actions,
        value,
        log_prob,
        RNNStates(new_ac_lstm_states, new_moa_lstm_states),
    )

  def calc_influence_reward(
      self, prev_action_logits, counterfactual_logits, visibility
  ):
    """
    Compute influence of this agent on other agents.
    :param prev_action_logits: Logits for the agent's own policy/actions at t-1
    :param counterfactual_logits: The counterfactual action logits for actions made by other
    agents at t.
    """
    # prev actions sollte basierend auf den Kommentaren in ssd die Aktionen des letzten
    # steps enthalten
    prev_agent_actions = self.prev_actions
    softmax = th.nn.Softmax()

    predicted_logits = gather_nd(
        params=counterfactual_logits, indices=prev_agent_actions
    )

    predicted_logits = th.reshape(
        predicted_logits, [-1, self.num_other_agents, self.num_actions]
    )
    predicted_logits = softmax(predicted_logits)
    predicted_logits = predicted_logits / th.sum(
        predicted_logits, dim=-1, keepdim=True
    )
    marginal_logits = self.marginalize_predictions(
        prev_action_logits, counterfactual_logits
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

    # Zero out influence for steps where the other agent isn't visible.
    if self.influence_only_when_visible:
      visibility = visibility.type(th.FloatTensor)
      influence_reward *= visibility

    influence_reward = th.sum(influence_reward, dim=-1)
    self.influence_reward = influence_reward

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
        [-1, self.num_actions, self.num_other_agents, self.num_actions],
    )
    counterfactual_logits = softmax(counterfactual_logits)

    # Change shape to broadcast probability of each action over counterfactual actions
    logits = th.reshape(logits, [-1, self.num_actions, 1, 1])

    normalized_counterfacts = logits * counterfactual_logits
    # Remove counterfactual action dimension
    marginal_probs = th.sum(normalized_counterfacts, dim=-3)
    # normalize
    marginal_probs = marginal_probs / th.sum(
        marginal_probs, dim=-1, keepdim=True
    )

    return marginal_probs

  # TODO: probably should add something like a predict moa Method, for calculating
  # the influence reward!
  def predict_values(
      self,
      obs: th.Tensor,
      lstm_states: Tuple[th.Tensor, th.Tensor],
      episode_starts: th.Tensor,
  ) -> th.Tensor:
    """
    Get the estimated values according to the current policy given the observations.

    :param obs: Observation.
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :return: the estimated values.
    """
    features = self.extract_features(obs)
    latent_ac, _ = self.mlp_extractor.forward(features)
    _, value, _ = self.ac_lstm.forward(latent_ac, lstm_states, episode_starts)

    return value

  def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
    """
    Evaluate actions according to the current policy,
    given the observations.

    :param obs: Observation.
    :param actions: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!probably only own actions!!!!!!
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :return: estimated value, log likelihood of taking those actions
        and entropy of the action distribution.
    """
    features = self.extract_features(obs)
    # ?????????
    latent_ac, latent_moa = self.mlp_extractor.forward(features)
    latent_pi, values, _ = self.ac_lstm.forward(
        latent_ac, lstm_states.ac, episode_starts
    )

    distribution = self._get_action_dist_from_latent(latent_pi)

    log_prob = distribution.log_prob(actions)

    return values, log_prob, distribution.entropy()

  # TODO: Setup_MOA_loss

  def calc_moa_loss(self, pred_actions, true_actions, loss_weight=1.0):
    """
    applying the approach to reward an agent for influencing others, we need
    to know the grade of influence. Thus we train the moa_part of the policy with
    the moa_loss. The implementation in ssd-games suggests modelling the loss by
    calculating the softmax cross entropy with the predicted actions and the true ones
    for each timestep, except the first
    In our case we will use the CrossEntropyLoss function from the torch lib,
    which combines LogSoftmax and negative-log-likelihood, instead of using plain
    softmax, which might be numerically less stable

    Args:
        pred_actions (th.Tensor): A Tensor containing the predicted actions for all agents
                                  for each timestep. Its dimensions are (B,N,A),
                                  where B is the number of timesteps, N the number of
                                  agents and A the number of actions
        true_actions (th.Tensor): A Tensor containing the true actions of each agent
                                  at each timestep. Its dimesnions are (B,N), where B
                                  is the number of timesteps (batchsize) and N the number
                                  of agents (considering, there will be one 'action' per agent
                                  instead of logits for all possible actions)
        loss_weight (float, optional): the weight used to calculate the total loss.
                                       Defaults to 1.0.
    """
    # There cannot be a prediction at t=-1 as well as there won't be an action after
    # the last batch, so we remove the first and last predictions to have only
    # predictions for sensible data
    action_logits = pred_actions[1:-1, :, :]

    # We need to adapt the shape of true_actions, so it fits action_logits
    # NOTE: ssd-games uses tf.cast() with tf.int32 on true_actions, though I currently
    #       don't see a good reason to do that as well
    true_actions = true_actions[2:, :]

    # compute the softmax cross_entropy
    loss_layer = th.nn.CrossEntropyLoss()
    ce_loss_per_entry = loss_layer(action_logits, true_actions)

    # NOTE: at this point ssd-games takes the visibility of agents for each other
    #       into consideration. We will not do that for now.

    total_loss = th.mean(ce_loss_per_entry)

    return total_loss, ce_loss_per_entry
