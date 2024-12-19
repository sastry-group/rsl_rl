#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrentMultiAgent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        super().__init__()

        self.rnn_hidden_size = rnn_hidden_size
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        activation = get_activation(activation)

        self.memory_a0 = Memory(num_actor_obs // 2, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_a1 = Memory(num_actor_obs // 2, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        self.mlp_a0 = self.build_mlp(rnn_hidden_size, actor_hidden_dims, num_actions // 2)
        self.mlp_a1 = self.build_mlp(rnn_hidden_size, actor_hidden_dims, num_actions - num_actions // 2)
        self.mlp_c = self.build_mlp(rnn_hidden_size, critic_hidden_dims, 1)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        self.memory_a0.reset(dones)
        self.memory_a1.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a0 = self.memory_a0(observations[..., :self.num_actor_obs // 2], masks, hidden_states)
        input_a1 = self.memory_a1(observations[..., self.num_actor_obs // 2:], masks, hidden_states)
        mean0 = self.mlp_a0(input_a0.squeeze(0))
        mean1 = self.mlp_a1(input_a1.squeeze(0))
        mean = torch.cat([mean0, mean1], dim=-1)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def act_inference(self, observations):
        hidden_states = self.get_actor_hidden_states()
        self.memory_a0.hidden_states = hidden_states
        self.memory_a1.hidden_states = hidden_states
        input_a0 = self.memory_a0(observations[:, :self.num_actor_obs // 2])
        input_a1 = self.memory_a1(observations[:, self.num_actor_obs // 2:])
        mean0 = self.mlp_a0(input_a0.squeeze(0))
        mean1 = self.mlp_a1(input_a1.squeeze(0))
        mean = torch.cat([mean0, mean1], dim=-1)
        return mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return self.mlp_c(input_c.squeeze(0))
    
    def get_actor_hidden_states(self):
        if self.memory_a0.hidden_states is None or self.memory_a1.hidden_states is None:
            return None
        a0_h, a0_c = self.memory_a0.hidden_states
        a1_h, a1_c = self.memory_a1.hidden_states
        
        h = torch.cat([a0_h[:, :, :self.rnn_hidden_size // 2], a1_h[:, :, (self.rnn_hidden_size // 2):]], dim=-1)
        c = torch.cat([a0_c[:, :, :self.rnn_hidden_size // 2], a1_c[:, :, (self.rnn_hidden_size // 2):]], dim=-1)

        return (h, c)

    def get_hidden_states(self):
        return self.get_actor_hidden_states(), self.memory_c.hidden_states

    def build_mlp(self, input_dim, hidden_dims, output_dim, activation="elu"):
        activation = get_activation(activation)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
