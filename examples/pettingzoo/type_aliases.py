from typing import NamedTuple, Tuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict


class RNNStates(NamedTuple):
    ac: Tuple[th.Tensor, ...]
    moa: Tuple[th.Tensor, ...]


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor


class RecurrentDictRolloutBufferSamples(NamedTuple):
  observations: TensorDict
  actions: th.Tensor
  old_values: th.Tensor
  old_log_prob: th.Tensor
  advantages: th.Tensor
  returns: th.Tensor
  lstm_states: RNNStates
  episode_starts: th.Tensor
  mask: th.Tensor


class MoaRolloutBufferSamples(NamedTuple):
  observations: th.Tensor
  actions: th.Tensor
  others_acts: th.Tensor
  old_values: th.Tensor
  old_log_prob: th.Tensor
  advantages: th.Tensor
  returns: th.Tensor
  pure_rews: th.Tensor
  lstm_states: RNNStates
  pred_actions: th.Tensor
  episode_starts: th.Tensor
  mask: th.Tensor
