import torch.nn as nn
import torch
from .torch_utils import  get_activation,check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 encoder_hidden_dims = [512, 256, 128, 64],
                 decoder_hidden_dims = [64, 128, 256, 512]):
        
        super(VAE, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent 

        # Build Encoder
        self.encoder = MLPHistoryEncoder(
            num_obs = num_obs,
            num_history=num_history,
            num_latent=num_latent*2,
            activation=activation,
            hidden_dims=encoder_hidden_dims,
        )
        # Build Encoder
        # self.encoder = MLPHistoryEncoder(
        #     num_obs = 825, ### TODO
        #     num_history=1,
        #     num_latent=num_latent*2,
        #     activation=activation,
        #     hidden_dims=encoder_hidden_dims,
        # )
        
        # Build Decoder
        modules = []
        activation_fn = nn.ELU()
        decoder_input_dim = num_latent
        modules.extend(
            [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
            activation_fn]
            )
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l],num_obs))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l],decoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.decoder = nn.Sequential(*modules)
    

    
    def encode(self,obs_history):
        '''  log_var
        '''
        encoded = self.encoder(obs_history)
        mu = encoded[...,:self.num_latent]
        var = encoded[...,self.num_latent:2*self.num_latent]
        return mu, var
        
    def decode(self, z):
        output = self.decoder(z)
        return output


    def forward(self,obs_history):
        mu, var = self.encode(obs_history)
        var_clipped = torch.clamp(var,-6.5,4.5) ###  var = (0.1, 5)
        z = self.reparameterize(mu, var_clipped)
        return z, mu, var_clipped
    
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar) ## log(var^2) -> std= exp(0.5*log(var^2))
        eps = torch.randn_like(std)  ### 标准正太分布
        return eps * std + mu
    

    def loss_fn(self, obs_history, next_obs, kld_weight = 1.0):
        z, mu, var = self.forward(obs_history)

        # Reconstruction loss
        recons = self.decode(z)
        recons_loss =F.mse_loss(recons, next_obs,reduction='none').mean(-1)
        # KL loss
        kld_loss = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = -1)

        loss = recons_loss + kld_weight * kld_loss

        return {
                    'loss': loss,
                    'recons_loss': recons_loss,
                    'kld_loss': kld_loss,
        }
    

    def sample(self,obs_history):
        z, _, _ = self.forward(obs_history)
        return z
    
    def inference(self,obs_history):
        _, mu, _ = self.forward(obs_history)
        return mu




class MlpEstimator(nn.Module):
    ''' num_obs: num_obs_step
        num_history: 
        num_latent: output_size
    '''
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_output,
                 hidden_dims = [256, 128]):
        super(MlpEstimator, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        input_size = num_obs * num_history
        output_size = num_output

        self.activation = nn.ELU()

        module_layers = []
        module_layers.append(nn.Linear(input_size, hidden_dims[0]))
        module_layers.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                module_layers.append(
                    nn.Linear(hidden_dims[l], output_size))
            else:
                module_layers.append(
                    nn.Linear(hidden_dims[l],
                              hidden_dims[l + 1]))
                module_layers.append(self.activation)
        self.model = nn.Sequential(*module_layers)
        self.model.train()
        print("Vel_est_model:",self.model)
        
        
    def forward(self, obs_history):
        output = self.model(obs_history)
        return output


class MLPHistoryEncoder(nn.Module):
    '''num_obs: num_obs_step
        num_history: 
        num_latent: output_size
    '''
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 hidden_dims = [256, 128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        input_size = num_obs * num_history
        output_size = num_latent

        self.activation = nn.ELU()

        # Adaptation module
        module_layers = []
        module_layers.append(nn.Linear(input_size, hidden_dims[0]))
        module_layers.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                module_layers.append(
                    nn.Linear(hidden_dims[l], output_size))
            else:
                module_layers.append(
                    nn.Linear(hidden_dims[l],
                              hidden_dims[l + 1]))
                module_layers.append(self.activation)
        self.encoder = nn.Sequential(*module_layers)
        self.encoder.train()
        print("MLPHistoryEncoder:",self.encoder)
        
        
    def forward(self, obs_history):
        output = self.encoder(obs_history)
        return output
        