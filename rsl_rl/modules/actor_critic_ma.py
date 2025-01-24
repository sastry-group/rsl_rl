from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

class MultiAgentActor(nn.Module):
    def __init__(self, num_agents, num_input_total, num_actions_total, num_history, actor_hidden_dims=[256, 256, 256], activation=None):
        super().__init__()
        self.num_agents = num_agents
        self.num_input_total = num_input_total
        self.num_actions_total = num_actions_total
        self.num_history = num_history
        self.actor_hidden_dims = actor_hidden_dims
        self.activation = activation
        self.num_input_per_agent = int(num_input_total // num_agents)
        self.num_actions_per_agent = int(num_actions_total // num_agents)

        assert num_input_total % num_agents == 0, "num_input_total must be divisible by num_agents"
        assert num_actions_total % num_agents == 0, "num_actions_total must be divisible by num_agents"

        actor_layers = []
        actor_layers.append(nn.Linear(int(num_input_total // num_agents * num_history), actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions_total // num_agents))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(self.activation)
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, observations):
        x = observations.reshape(observations.shape[0], observations.shape[1], self.num_agents, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(start_dim=2)
        x = self.actor(x)
        x = x.reshape(x.shape[0], -1)
        return x
        # for i in range(self.num_agents):
        #     agent_observations = observations[:, :, i * self.num_input_per_agent:(i + 1) * self.num_input_per_agent].flatten(start_dim=1)
        #     agent_actions = self.actor(agent_observations)
        #     if i == 0:
        #         actions = agent_actions
        #     else:
        #         actions = torch.cat((actions, agent_actions), dim=-1)
        # return actions

class MultiAgentCritic(nn.Module):
    def __init__(self, num_input_total, num_history, critic_hidden_dims=[256, 256, 256], activation=None):
        super().__init__()
        self.num_input_total = num_input_total
        self.num_history = num_history
        self.critic_hidden_dims = critic_hidden_dims
        self.activation = activation

        critic_layers = []
        critic_layers.append(nn.Linear(num_input_total * num_history, critic_hidden_dims[0]))
        critic_layers.append(self.activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(self.activation)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, observations):
        x = observations.flatten(start_dim=1)
        return self.critic(x)

class ActorCriticMultiAgent(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        actor_obs_history_dim, actor_obs_dim = num_actor_obs
        critic_obs_history_dim, critic_obs_dim = num_critic_obs

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        assert actor_obs_dim % 2 == 0, "actor_obs_dim must be even"

        mlp_input_dim_c = int(critic_obs_dim * critic_obs_history_dim)
        # Policy
        self.actor = MultiAgentActor(2, actor_obs_dim, num_actions, actor_obs_history_dim, actor_hidden_dims, activation)

        # Value function
        self.critic = MultiAgentCritic(critic_obs_dim, critic_obs_history_dim, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None