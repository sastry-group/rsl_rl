#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchtune.modules import RotaryPositionalEmbeddings
from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.transformer import Transformer
from rsl_rl.utils import unpad_trajectories

class ActorCriticTransformer(nn.Module):
    is_recurrent = True
    is_transformer = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        transformer_hidden_size=256,
        transformer_num_layers=1,
        transformer_num_heads=8,
        context_len=16,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__()
        # Policy
        self.actor = Memory(num_actor_obs, num_actions, context_len, transformer_hidden_size, transformer_num_heads, transformer_num_layers)

        # Value function
        self.critic = Memory(num_critic_obs, 1, context_len, transformer_hidden_size, transformer_num_heads, transformer_num_layers)

        print(f"Actor Transformer: {self.actor}")
        print(f"Critic Transformer: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        print(f"Number of transformer actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Number of transformer critic parameters: {sum(p.numel() for p in self.critic.parameters())}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        raise NotImplementedError

    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)

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

    def act(self, observations, masks=None, hidden_states=None):
        mean = self.actor(observations, masks, hidden_states)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        value = self.critic(critic_observations, masks, hidden_states)
        return value

    def get_hidden_states(self):
        def get_past_key_values(actor):
            if actor.transformer.past_key_values is None:
                return None, None, None
            past_key_values = actor.transformer.past_key_values # List[(past_key, past_value)]
            past_keys = [past_key_values[i][0] for i in range(len(past_key_values))]
            past_keys = torch.stack(past_keys)
            past_values = [past_key_values[i][1] for i in range(len(past_key_values))]
            past_values = torch.stack(past_values)
            counter_arr = torch.full((1, past_keys.shape[1], 1, 1, 1), actor.counter)
            return past_keys, past_values, counter_arr
        return get_past_key_values(self.actor), get_past_key_values(self.critic)

class Memory(torch.nn.Module):
    def __init__(self, num_actor_obs, num_actions, context_len, transformer_hidden_size, transformer_num_heads, transformer_num_layers):
        super().__init__()
        self.transformer = Transformer(num_actor_obs, num_actions, context_len, transformer_hidden_size, transformer_num_heads, transformer_num_layers)
        self.counter = 0

    def forward(self, input, masks=None, hidden_states=None):
        # transformer hidden states are the past keys and values of each layer
        # hidden states is (past_key, past_value)
        # past_key and past_value: (layer_num, batch_size, num_heads, seq_len, head_dim)
        # input: (horizon_len, batch_size, num_obs)

        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            # change hidden state to what the transformer expects
            # which is List[(past_key, past_value)] where each item is for a layer
            # past_key and past_value: (batch_size, num_heads, seq_len, head_dim)
            past_key, past_value, stored_counter = hidden_states
            past_key_values = []
            for layer_num in range(past_key.shape[0]):
                past_key_values.append((past_key[layer_num], past_value[layer_num]))
            old_past_key_values = self.transformer.past_key_values
            old_counter = self.counter
            self.transformer.past_key_values = past_key_values
            self.counter = stored_counter.view(-1)
            out = []
            for i in range(len(input)):
                # go through each step in the horizon
                out.append(self.transformer(input[i].unsqueeze(1), use_cache=True, update_cache=True, position_step=self.counter)[:, -1, :])
                self.counter += 1
            out = torch.stack(out)
            out = unpad_trajectories(out, masks)
            self.transformer.past_key_values = old_past_key_values
            self.counter = old_counter
        else:
            # inference mode (collection): use hidden states of last step
            out = self.transformer(input[:, None, :], use_cache=True, update_cache=True, position_step=self.counter)[:, -1, :]
            self.counter += 1
        return out

    def reset(self, dones=None):
        for layer_kv in self.transformer.past_key_values:
            for kv in layer_kv:
                kv[dones.bool()] = 0.0 # type: ignore
