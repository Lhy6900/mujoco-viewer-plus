import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .state_estimator import VAE
from .torch_utils import init_orhtogonal


class ActorCritic(nn.Module):
    def __init__(self,  num_vae,
                        num_obs_step,
                        num_critic_obs,
                        num_history,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0
                        ):
        super(ActorCritic, self).__init__()
        activation = get_activation(activation)

        ### actor,  critic ---------------------
        mlp_input_dim_a = num_obs_step + num_vae
        mlp_input_dim_c = num_critic_obs

        # Policy actor
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)

        self.actor = nn.Sequential(*actor_layers)

        # Value function     critic
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        self.vae = VAE(
            num_obs= num_obs_step,
            num_history= num_history,
            num_latent= num_vae,
            activation=activation,
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")


        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        



    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    ### 没用的函数
    def reset(self, dones=None):
        # print("AC_reset========================")
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
        self.distribution = Normal(mean, mean*0. + torch.clamp(self.std, min=1e-3))

    def act(self, obs,  obs_history):
        latent = self.vae.sample(obs_history)
        actor_obs = torch.cat((latent, obs), dim = -1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()
   

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_history):
        latent = self.vae.inference(obs_history)
        actor_obs = torch.cat((latent, obs), dim = -1)
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    
    def evaluate(self, critic_observations):
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
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
