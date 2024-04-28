import numpy as np
import torch as th

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

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
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        fc_activation_fn: Type[nn.Module] = nn.Tanh,
        conv_activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.conv_activation_fn = conv_activation_fn

        super.__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=fc_activation_fn,
            share_features_extractor=share_features_extractor,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class = optimizer_class,
            optimizer_kwargs = optimizer_kwargs,
            normalize_images = normalize_images 
        )
        
     def build(lr_schedule):
        print("Hello")

    