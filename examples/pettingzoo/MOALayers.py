from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.utils import get_device, zip_strict


class MOAMlp(nn.Module):
  """
  Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
  the observations (if no features extractor is applied) as an input and outputs a latent representation
  for the Actor critic and MOA Network

  The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
  It should be in the following form:
  ``dict(conv=[<list of tuples of the form (layer size, kernel_size)>], ac=[<list of layer sizes>], moa=[<list of layer sizes>])``: to specify
      the amount and size of the layers in the shared convolutional Part, ActorCritc and MOA nets individually.
      If it is missing any of the keys (conv, ac or moa), a default .


  .. note::
      If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

  :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
  :param net_arch: The specification of the networks.
      See above for details on its formatting.
  :param fc_activation_fn: The activation function to use for the fully connected layers.
  :param conv_activation_fn:
  :param device: PyTorch device.
  """

  def __init__(
      self,
      feature_dim: int,
      net_arch: Dict[str, List],
      conv_activation_fn: Type[nn.Module],
      ac_activation_fn: Type[nn.Module],
      moa_activation_fn: Type[nn.Module],
      device: Union[th.device, str] = "auto",
  ) -> None:
    super.__init__()
    device = get_device(device)

    conv_net_arch = net_arch.get("conv", None)
    ac_net_arch = net_arch.get("ac", None)
    moa_net_arch = net_arch.get("moa", None)

    self.conv_layers, conv_out_dim = self.build_conv_layers(
        feature_dim, conv_net_arch, activation_fn=conv_activation_fn
    )

    self.ac_fc_layers, self.ac_out_dim = self.build_fc_layers(
        conv_out_dim, ac_net_arch, activation_fn=ac_activation_fn
    )

    self.moa_fc_layers, self.moa_out_dim = self.build_fc_layers(
        conv_out_dim, moa_net_arch, activation_fn=moa_activation_fn
    )

  def build_conv_layers(self, in_size, net_arch, activation_fn=nn.ReLU):
    if net_arch == None:
      net_arch = [(6, (3, 3))]

    conv_layers = []
    for out_size, kernel_size in net_arch:
      conv_layers.append(nn.Conv2d(in_size, out_size, kernel_size))
      conv_layers.append(activation_fn)
      in_size = out_size

    return nn.Sequential(conv_layers), in_size

  def build_fc_layers(self, in_size, out_sizes, activation_fn=nn.Tanh):
    if out_sizes == None:
      out_sizes = [32, 32]

    fc_layers = []
    for out_size in out_sizes:
      fc_layers.append(nn.Conv2d(in_size, out_size))
      fc_layers.append(activation_fn)
      in_size = out_size

    return nn.Sequential(fc_layers), in_size

  def forward(self, obs):
    conv_output = self.conv_layers(obs)
    return self.ac_fc_layers(conv_output), self.moa_fc_layers(conv_output)

  def get_ac_out_dim(self):
    return self.ac_out_dim

  def get_moa_out_dim(self):
    return self.moa_out_dim


class ACLSTM(nn.Module):

  def __init__(
      self,
      in_size,
      cell_size,
      action_out_size=32,
      num_lstm_layers=1,
      lstm_kwargs={},
  ):

    super().__init__()

    self.action_out_size = action_out_size

    self.lstm = nn.LSTM(
        in_size, cell_size, num_layers=num_lstm_layers, **lstm_kwargs
    )
    self.logit_layer = nn.Linear(self.lstm.hidden_size, action_out_size)
    self.value_out = nn.Linear(self.lstm.hidden_size, 1)

  def forward(self, features, lstm_states, episode_starts):

    lstm_output, new_lstm_states = self._process_sequence(
        features, lstm_states, episode_starts
    )

    logits = self.logit_layer(lstm_output)
    value_out = self.value_out(lstm_output)

    return logits, value_out, new_lstm_states

  def _process_sequence(
      self,
      features: th.Tensor,
      lstm_states: Tuple[th.Tensor, th.Tensor],
      episode_starts: th.Tensor,
  ) -> Tuple[th.Tensor, th.Tensor]:
    """
    Do a forward pass in the LSTM network.

    :param features: Input tensor
    :param lstm_states: previous hidden and cell states of the LSTM, respectively
    :param episode_starts: Indicates when a new episode starts,
        in that case, we need to reset LSTM states.
    :param lstm: LSTM object.
    :return: LSTM output and updated LSTM states.
    """
    # LSTM logic
    # (sequence length, batch size, features dim)
    # (batch size = n_envs for data collection or n_seq when doing gradient update)
    n_seq = lstm_states[0].shape[1]
    # Batch to sequence
    # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
    # note: max length (max sequence length) is always 1 during data collection
    features_sequence = features.reshape(
        (n_seq, -1, self.lstm.input_size)
    ).swapaxes(0, 1)
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    # If we don't have to reset the state in the middle of a sequence
    # we can avoid the for loop, which speeds up things
    if th.all(episode_starts == 0.0):
      lstm_output, lstm_states = self.lstm(features_sequence, lstm_states)
      lstm_output = th.flatten(
          lstm_output.transpose(0, 1), start_dim=0, end_dim=1
      )
      return lstm_output, lstm_states

    lstm_output = []
    # Iterate over the sequence
    for features, episode_start in zip_strict(
        features_sequence, episode_starts
    ):
      hidden, lstm_states = self.lstm(
          features.unsqueeze(dim=0),
          (
              # Reset the states at the beginning of a new episode
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
          ),
      )
      lstm_output += [hidden]
    # Sequence to batch
    # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
    lstm_output = th.flatten(
        th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
    )
    return lstm_output, lstm_states

  def get_act_out_size(self):
    return self.action_out_size


class MOALSTM(nn.Module):

  def __init__(
      self,
      num_features,
      num_outputs,
      num_actions,
      cell_size,
      num_lstm_layers=1,
      lstm_kwargs={},
  ) -> None:
    """
    :param num_features the size of the tensor containing the information regarding the observations,
           can be the output size of other Layers in the NN or the the size of the original observation
           tensor
    :param num_logits the size of the Tensor containing the actions
           of all other agents (so num_agents x num_actions), which is also the output size
    :param num_actions the size of the Tensor containing the number of action, to handle the agents own actions
    :param cell_size the amount of LSTM units
    :param num_lstm_layers the number of layers for the lstm
    :param lstm_kwargs nn.LSTM allows further args
    """
    super().__init__()

    self.cell_size = cell_size
    self.in_size = num_features + num_outputs + num_actions

    self.lstm = nn.LSTM(
        self.in_size, cell_size, num_layers=num_lstm_layers, **lstm_kwargs
    )
    self.logits = nn.Linear(self.lstm.hidden_size, num_outputs)

  def forward(self, features, lstm_states, episode_starts):

    lstm_output, new_lstm_states = self._process_sequence(
        features, lstm_states, episode_starts
    )

    logits = self.logits(lstm_output)

    return logits, new_lstm_states

  def _process_sequence(
      self,
      features: th.Tensor,
      lstm_states: Tuple[th.Tensor, th.Tensor],
      episode_starts: th.Tensor,
  ) -> Tuple[th.Tensor, th.Tensor]:
    """
    Do a forward pass in the LSTM network.

    :param features: Input tensor
    :param lstm_states: previous hidden and cell states of the LSTM, respectively
    :param episode_starts: Indicates when a new episode starts,
        in that case, we need to reset LSTM states.
    :param lstm: LSTM object.
    :return: LSTM output and updated LSTM states.
    """
    # LSTM logic
    # (sequence length, batch size, features dim)
    # (batch size = n_envs for data collection or n_seq when doing gradient update)
    n_seq = lstm_states[0].shape[1]
    # Batch to sequence
    # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
    # note: max length (max sequence length) is always 1 during data collection
    features_sequence = features.reshape(
        (n_seq, -1, self.lstm.input_size)
    ).swapaxes(0, 1)
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    # If we don't have to reset the state in the middle of a sequence
    # we can avoid the for loop, which speeds up things
    if th.all(episode_starts == 0.0):
      lstm_output, lstm_states = self.lstm(features_sequence, lstm_states)
      lstm_output = th.flatten(
          lstm_output.transpose(0, 1), start_dim=0, end_dim=1
      )
      return lstm_output, lstm_states

    lstm_output = []
    # Iterate over the sequence
    for features, episode_start in zip_strict(
        features_sequence, episode_starts
    ):
      hidden, lstm_states = self.lstm(
          features.unsqueeze(dim=0),
          (
              # Reset the states at the beginning of a new episode
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
          ),
      )
      lstm_output += [hidden]
    # Sequence to batch
    # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
    lstm_output = th.flatten(
        th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
    )
    return lstm_output, lstm_states
