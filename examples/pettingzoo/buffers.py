from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from type_aliases import (
    RecurrentDictRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    RNNStates,
    MoaRolloutBufferSamples,
)


def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
  """
  Chunk sequences and pad them to have constant dimensions.

  :param seq_start_indices: Indices of the transitions that start a sequence
  :param seq_end_indices: Indices of the transitions that end a sequence
  :param device: PyTorch device
  :param tensor: Tensor of shape (batch_size, *tensor_shape)
  :param padding_value: Value used to pad sequence to the same length
      (zero padding by default)
  :return: (n_seq, max_length, *tensor_shape)
  """
  # Create sequences given start and end
  seq = [
      th.tensor(tensor[start : end + 1], device=device)
      for start, end in zip(seq_start_indices, seq_end_indices)
  ]
  return th.nn.utils.rnn.pad_sequence(
      seq, batch_first=True, padding_value=padding_value
  )


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
  """
  Pad and flatten the sequences of scalar values,
  while keeping the sequence order.
  From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

  :param seq_start_indices: Indices of the transitions that start a sequence
  :param seq_end_indices: Indices of the transitions that end a sequence
  :param device: PyTorch device (cpu, gpu, ...)
  :param tensor: Tensor of shape (max_length, n_seq, 1)
  :param padding_value: Value used to pad sequence to the same length
      (zero padding by default)
  :return: (n_seq * max_length,) aka (padded_batch_size,)
  """
  return pad(
      seq_start_indices, seq_end_indices, device, tensor, padding_value
  ).flatten()


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
) -> Tuple[np.ndarray, Callable, Callable]:
  """
  Create the utility function to chunk data into
  sequences and pad them to create fixed size tensors.

  :param episode_starts: Indices where an episode starts
  :param env_change: Indices where the data collected
      come from a different env (when using multiple env for data collection)
  :param device: PyTorch device
  :return: Indices of the transitions that start a sequence,
      pad and pad_and_flatten utilities tailored for this batch
      (sequence starts and ends indices are fixed)
  """
  # Create sequence if env changes too
  seq_start = np.logical_or(episode_starts, env_change).flatten()
  # First index is always the beginning of a sequence
  seq_start[0] = True
  # Retrieve indices of sequence starts
  seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
  # End of sequence are just before sequence starts
  # Last index is also always end of a sequence
  seq_end_indices = np.concatenate(
      [(seq_start_indices - 1)[1:], np.array([len(episode_starts)])]
  )

  # Create padding method for this minibatch
  # to avoid repeating arguments (seq_start_indices, seq_end_indices)
  local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
  local_pad_and_flatten = partial(
      pad_and_flatten, seq_start_indices, seq_end_indices, device
  )
  return seq_start_indices, local_pad, local_pad_and_flatten


class RecurrentRolloutBuffer(RolloutBuffer):
  """
  Rollout buffer that also stores the LSTM cell and hidden states.

  :param buffer_size: Max number of element in the buffer
  :param observation_space: Observation space
  :param action_space: Action space
  :param hidden_state_shape: Shape of the buffer that will collect lstm states
      (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
  :param device: PyTorch device
  :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
      Equivalent to classic advantage when set to 1.
  :param gamma: Discount factor
  :param n_envs: Number of parallel environments
  """

  def __init__(
      self,
      buffer_size: int,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      hidden_state_shape: Tuple[int, int, int, int],
      device: Union[th.device, str] = "auto",
      gae_lambda: float = 1,
      gamma: float = 0.99,
      n_envs: int = 1,
  ):
    self.hidden_state_shape = hidden_state_shape
    self.seq_start_indices, self.seq_end_indices = None, None
    super().__init__(
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda,
        gamma,
        n_envs,
    )

  def reset(self):
    super().reset()
    self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

  def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
    """
    :param hidden_states: LSTM cell and hidden state
    """
    self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
    self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
    self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
    self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

    super().add(*args, **kwargs)

  def get(
      self, batch_size: Optional[int] = None
  ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
    assert self.full, "Rollout buffer must be full before sampling from it"

    # Prepare the data
    if not self.generator_ready:
      # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
      # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
      for tensor in [
          "hidden_states_pi",
          "cell_states_pi",
          "hidden_states_vf",
          "cell_states_vf",
      ]:
        self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

      # flatten but keep the sequence order
      # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
      # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
      for tensor in [
          "observations",
          "actions",
          "values",
          "log_probs",
          "advantages",
          "returns",
          "hidden_states_pi",
          "cell_states_pi",
          "hidden_states_vf",
          "cell_states_vf",
          "episode_starts",
      ]:
        self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
      self.generator_ready = True

    # Return everything, don't create minibatches
    if batch_size is None:
      batch_size = self.buffer_size * self.n_envs

    # Sampling strategy that allows any mini batch size but requires
    # more complexity and use of padding
    # Trick to shuffle a bit: keep the sequence order
    # but split the indices in two
    split_index = np.random.randint(self.buffer_size * self.n_envs)
    indices = np.arange(self.buffer_size * self.n_envs)
    indices = np.concatenate((indices[split_index:], indices[:split_index]))

    env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
        self.buffer_size, self.n_envs
    )
    # Flag first timestep as change of environment
    env_change[0, :] = 1.0
    env_change = self.swap_and_flatten(env_change)

    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
      batch_inds = indices[start_idx : start_idx + batch_size]
      yield self._get_samples(batch_inds, env_change)
      start_idx += batch_size

  def _get_samples(
      self,
      batch_inds: np.ndarray,
      env_change: np.ndarray,
      env: Optional[VecNormalize] = None,
  ) -> RecurrentRolloutBufferSamples:
    # Retrieve sequence starts and utility function
    self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
        self.episode_starts[batch_inds], env_change[batch_inds], self.device
    )

    # Number of sequences
    n_seq = len(self.seq_start_indices)
    max_length = self.pad(self.actions[batch_inds]).shape[1]
    padded_batch_size = n_seq * max_length
    # We retrieve the lstm hidden states that will allow
    # to properly initialize the LSTM at the beginning of each sequence
    lstm_states_pi = (
        # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
        # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
        # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(
            0, 1
        ),
        self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
    )
    lstm_states_vf = (
        # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(
            0, 1
        ),
        self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
    )
    lstm_states_pi = (
        self.to_torch(lstm_states_pi[0]).contiguous(),
        self.to_torch(lstm_states_pi[1]).contiguous(),
    )
    lstm_states_vf = (
        self.to_torch(lstm_states_vf[0]).contiguous(),
        self.to_torch(lstm_states_vf[1]).contiguous(),
    )

    return RecurrentRolloutBufferSamples(
        # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
        observations=self.pad(self.observations[batch_inds]).reshape(
            (padded_batch_size, *self.obs_shape)
        ),
        actions=self.pad(self.actions[batch_inds]).reshape(
            (padded_batch_size,) + self.actions.shape[1:]
        ),
        old_values=self.pad_and_flatten(self.values[batch_inds]),
        old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
        advantages=self.pad_and_flatten(self.advantages[batch_inds]),
        returns=self.pad_and_flatten(self.returns[batch_inds]),
        lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
        episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
        mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
    )


class RecurrentDictRolloutBuffer(DictRolloutBuffer):
  """
  Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
  Extends the RecurrentRolloutBuffer to use dictionary observations

  :param buffer_size: Max number of element in the buffer
  :param observation_space: Observation space
  :param action_space: Action space
  :param hidden_state_shape: Shape of the buffer that will collect lstm states
  :param device: PyTorch device
  :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
      Equivalent to classic advantage when set to 1.
  :param gamma: Discount factor
  :param n_envs: Number of parallel environments
  """

  def __init__(
      self,
      buffer_size: int,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      hidden_state_shape: Tuple[int, int, int, int],
      device: Union[th.device, str] = "auto",
      gae_lambda: float = 1,
      gamma: float = 0.99,
      n_envs: int = 1,
  ):
    self.hidden_state_shape = hidden_state_shape
    self.seq_start_indices, self.seq_end_indices = None, None
    super().__init__(
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda,
        gamma,
        n_envs=n_envs,
    )

  def reset(self):
    super().reset()
    self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
    self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

  def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
    """
    :param hidden_states: LSTM cell and hidden state
    """
    self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
    self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
    self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
    self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

    super().add(*args, **kwargs)

  def get(
      self, batch_size: Optional[int] = None
  ) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
    assert self.full, "Rollout buffer must be full before sampling from it"

    # Prepare the data
    if not self.generator_ready:
      # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
      # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
      for tensor in [
          "hidden_states_pi",
          "cell_states_pi",
          "hidden_states_vf",
          "cell_states_vf",
      ]:
        self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

      for key, obs in self.observations.items():
        self.observations[key] = self.swap_and_flatten(obs)

      for tensor in [
          "actions",
          "values",
          "log_probs",
          "advantages",
          "returns",
          "hidden_states_pi",
          "cell_states_pi",
          "hidden_states_vf",
          "cell_states_vf",
          "episode_starts",
      ]:
        self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
      self.generator_ready = True

    # Return everything, don't create minibatches
    if batch_size is None:
      batch_size = self.buffer_size * self.n_envs

    # Trick to shuffle a bit: keep the sequence order
    # but split the indices in two
    split_index = np.random.randint(self.buffer_size * self.n_envs)
    indices = np.arange(self.buffer_size * self.n_envs)
    indices = np.concatenate((indices[split_index:], indices[:split_index]))

    env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
        self.buffer_size, self.n_envs
    )
    # Flag first timestep as change of environment
    env_change[0, :] = 1.0
    env_change = self.swap_and_flatten(env_change)

    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
      batch_inds = indices[start_idx : start_idx + batch_size]
      yield self._get_samples(batch_inds, env_change)
      start_idx += batch_size

  def _get_samples(
      self,
      batch_inds: np.ndarray,
      env_change: np.ndarray,
      env: Optional[VecNormalize] = None,
  ) -> RecurrentDictRolloutBufferSamples:
    # Retrieve sequence starts and utility function
    self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
        self.episode_starts[batch_inds], env_change[batch_inds], self.device
    )

    n_seq = len(self.seq_start_indices)
    max_length = self.pad(self.actions[batch_inds]).shape[1]
    padded_batch_size = n_seq * max_length
    # We retrieve the lstm hidden states that will allow
    # to properly initialize the LSTM at the beginning of each sequence
    lstm_states_pi = (
        # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(
            0, 1
        ),
        self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
    )
    lstm_states_vf = (
        # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(
            0, 1
        ),
        self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
    )
    lstm_states_pi = (
        self.to_torch(lstm_states_pi[0]).contiguous(),
        self.to_torch(lstm_states_pi[1]).contiguous(),
    )
    lstm_states_vf = (
        self.to_torch(lstm_states_vf[0]).contiguous(),
        self.to_torch(lstm_states_vf[1]).contiguous(),
    )

    observations = {
        key: self.pad(obs[batch_inds])
        for (key, obs) in self.observations.items()
    }
    observations = {
        key: obs.reshape((padded_batch_size,) + self.obs_shape[key])
        for (key, obs) in observations.items()
    }

    return RecurrentDictRolloutBufferSamples(
        observations=observations,
        actions=self.pad(self.actions[batch_inds]).reshape(
            (padded_batch_size,) + self.actions.shape[1:]
        ),
        old_values=self.pad_and_flatten(self.values[batch_inds]),
        old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
        advantages=self.pad_and_flatten(self.advantages[batch_inds]),
        returns=self.pad_and_flatten(self.returns[batch_inds]),
        lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
        episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
        mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
    )


class MOABuffer(RolloutBuffer):
  """
  Recurrent Rollout buffer that also stores the LSTM cell and hidden states for
  the MOA part of the model of a MOA-AC agent. As this class is also created
  with decentralized training in mind,
  NOTE: based on the RecurrentRolloutBuffer implementation from sb3-contrib

  :param buffer_size: Max number of element in the buffer
  :param observation_space: Observation space
  :param action_space: Action space
  :param hidden_state_shape: Shape of the buffer that will collect lstm states
      (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
  :param device: PyTorch device
  :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
      Equivalent to classic advantage when set to 1.
  :param gamma: Discount factor
  :param n_envs: Number of parallel environments
  """

  def __init__(
      self,
      buffer_size: int,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      ac_hidden_state_shape: Tuple[int, int, int, int],
      moa_hidden_state_shape: Tuple[int, int, int, int],
      device: Union[th.device, str] = "auto",
      gae_lambda: float = 1,
      gamma: float = 0.99,
      n_envs: int = 1,
  ):
    self.ac_hidden_state_shape = ac_hidden_state_shape
    self.moa_hidden_state_shape = moa_hidden_state_shape
    self.seq_start_indices, self.seq_end_indices = None, None

    super().__init__(
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda,
        gamma,
        n_envs,
    )

  def reset(self):
    super().reset()
    self.pure_rews = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
    self.hidden_states_ac = np.zeros(
        self.ac_hidden_state_shape, dtype=np.float32
    )
    self.cell_states_ac = np.zeros(self.ac_hidden_state_shape, dtype=np.float32)
    self.hidden_states_moa = np.zeros(
        self.moa_hidden_state_shape, dtype=np.float32
    )
    self.cell_states_moa = np.zeros(
        self.moa_hidden_state_shape, dtype=np.float32
    )
    self.pred_actions = np.zeros(
        (self.buffer_size, self.n_envs, self.action_dim * self.n_envs),
        dtype=np.float32,
    )

  def add(
      self,
      *args,
      lstm_states: RNNStates,
      num_agents,
      pred_actions,
      pure_rews,
      **kwargs
  ) -> None:
    """
    :param hidden_states: LSTM cell and hidden state
    """
    for agent in range(num_agents):
      self.hidden_states_ac[self.pos][agent] = np.array(
          lstm_states[agent].ac[0].cpu().numpy()
      )
      self.cell_states_ac[self.pos][agent] = np.array(
          lstm_states[agent].ac[1].cpu().numpy()
      )
      self.hidden_states_moa[self.pos][agent] = np.array(
          lstm_states[agent].moa[0].cpu().numpy()
      )
      self.cell_states_moa[self.pos][agent] = np.array(
          lstm_states[agent].moa[1].cpu().numpy()
      )
    self.pred_actions[self.pos] = np.array(pred_actions)
    self.pure_rews[self.pos] = np.array(pure_rews)

    super().add(*args, **kwargs)

    # TODO: adapt to the same agent performing in different envs, especially in regards to the lstms

  def get(
      self, agent: int, batch_size: Optional[int] = None
  ) -> Generator[RecurrentRolloutBufferSamples, None, None]:
    assert self.full, "Rollout buffer must be full before sampling from it"

    # Prepare the data
    if not self.generator_ready:
      # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
      # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
      for tensor in [
          "hidden_states_ac",
          "cell_states_ac",
          "hidden_states_moa",
          "cell_states_moa",
      ]:
        self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

      # flatten but keep the sequence order
      # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
      # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
      for tensor in [
          "observations",
          "actions",
          "values",
          "log_probs",
          "advantages",
          "returns",
          "hidden_states_ac",
          "cell_states_ac",
          "hidden_states_moa",
          "cell_states_moa",
          "episode_starts",
      ]:
        self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
      self.generator_ready = True

    # Return everything, don't create minibatches
    if batch_size is None:
      batch_size = self.buffer_size * self.n_envs

    # Sampling strategy that allows any mini batch size but requires
    # more complexity and use of padding
    # Trick to shuffle a bit: keep the sequence order
    # but split the indices in two
    split_index = np.random.randint(self.buffer_size)
    indices = np.arange(self.buffer_size)
    indices = np.concatenate((indices[split_index:], indices[:split_index]))

    env_change = np.zeros(self.buffer_size * self.n_envs).reshape(
        self.buffer_size, self.n_envs
    )
    # Flag first timestep as change of environment
    env_change[0, :] = 1.0
    env_change = self.swap_and_flatten(env_change)

    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
      batch_inds = indices[start_idx : start_idx + batch_size]
      yield self._get_samples(batch_inds, env_change, agent)
      start_idx += batch_size

  def _get_samples(
      self,
      batch_inds: np.ndarray,
      env_change: np.ndarray,
      agent: int,
      # env: Optional[VecNormalize] = None, NOTE: this parameter appears in the code from sb3-contrib, but its not used
      #                                           I'll leave it here as a comment, as it might be of use later
  ) -> RecurrentRolloutBufferSamples:
    # Retrieve sequence starts and utility function
    self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
        self.episode_starts[batch_inds], env_change[batch_inds], self.device
    )

    # TODO: fix padded batch size to fit the shape of the nd-arrays
    # Number of sequences
    n_seq = len(self.seq_start_indices)
    max_length = self.pad(self.actions[batch_inds]).shape[1]
    padded_batch_size = n_seq * max_length
    # We retrieve the lstm hidden states that will allow
    # to properly initialize the LSTM at the beginning of each sequence
    lstm_states_ac = (
        # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
        # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
        # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_ac[[batch_inds], agent][
            self.seq_start_indices
        ].swapaxes(0, 1),
        self.cell_states_ac[[batch_inds], agent][
            self.seq_start_indices
        ].swapaxes(0, 1),
    )
    lstm_states_moa = (
        # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
        self.hidden_states_moa[[batch_inds], agent][
            self.seq_start_indices
        ].swapaxes(0, 1),
        self.cell_states_moa[[batch_inds], agent][
            self.seq_start_indices
        ].swapaxes(0, 1),
    )
    lstm_states_ac = (
        self.to_torch(lstm_states_ac[0]).contiguous(),
        self.to_torch(lstm_states_moa[1]).contiguous(),
    )
    lstm_states_moa = (
        self.to_torch(lstm_states_ac[0]).contiguous(),
        self.to_torch(lstm_states_moa[1]).contiguous(),
    )

    return MoaRolloutBufferSamples(
        # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
        observations=self.pad(self.observations[[batch_inds], agent]).reshape(
            (padded_batch_size, *self.obs_shape)
        ),
        actions=self.pad(self.actions[[batch_inds], agent]).reshape(
            (padded_batch_size,) + self.actions.shape[1:]
        ),
        old_values=self.pad_and_flatten(self.values[[batch_inds], agent]),
        old_log_prob=self.pad_and_flatten(self.log_probs[[batch_inds], agent]),
        advantages=self.pad_and_flatten(self.advantages[[batch_inds], agent]),
        returns=self.pad_and_flatten(self.returns[[batch_inds], agent]),
        pure_rews=self.pad_and_flatten(self.pure_rews[[batch_inds], agent]),
        lstm_states=RNNStates(lstm_states_ac, lstm_states_moa),
        pred_actions=self.pad(self.pred_actions[[batch_inds], agent]),
        episode_starts=self.pad_and_flatten(self.episode_starts),
        mask=self.pad_and_flatten(
            np.ones_like(self.returns[[batch_inds], agent])
        ),
    )

  def compute_returns_and_advantage(
      self, last_values: th.Tensor, dones: np.ndarray, num_agents: int
  ) -> None:
    """
    post_processing: apply agents_returns_and_advantage to all agents to
    calculate the full returns and advantages

    Args:
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        :param num_agents: the number of agents for use of iteration
    """
    for agent in range(num_agents):
      self.agents_returns_and_advantage(last_values, dones, agent)

  def agents_returns_and_advantage(
      self, last_values: th.Tensor, dones: np.ndarray, agent: int
  ) -> None:
    """
    Post-processing step: compute the lambda-return(TD(lambda) estimate and
    GAE(lambda) advantage for a single agent

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
    where R is the sum of discounted reward with value bootstrap
    (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

    The TD(lambda) estimator has also two special cases:
    - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
    - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

    For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

    :param last_values: state value estimation for the last step (one for each env)
    :param dones: if the last step was a terminal step (one bool for each env).
    :param agent: the index of the agent to handle
    """
    # Convert to numpy
    last_values = last_values[agent].clone().cpu().numpy().flatten()

    last_gae_lam = 0
    for step in reversed(range(self.buffer_size)):
      if step == self.buffer_size - 1:
        next_non_terminal = 1.0 - dones[agent]
        next_values = last_values
      else:
        next_non_terminal = 1.0 - self.episode_starts[step + 1][agent]
        next_values = self.values[step + 1][agent]
      delta = (
          self.rewards[step][agent]
          + self.gamma * next_values * next_non_terminal
          - self.values[step][agent]
      )
      last_gae_lam = (
          delta
          + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
      )
      self.advantages[step][agent] = last_gae_lam
    # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
    # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
    self.returns[:, agent] = self.advantages[:, agent] + self.values[:, agent]
