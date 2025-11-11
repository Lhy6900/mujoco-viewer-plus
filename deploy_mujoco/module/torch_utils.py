import torch.nn as nn
import torch
from torch.distributions import Normal
from abc import ABC, abstractmethod
from typing import Tuple, Union
import json 
import numpy as np 

# minimal interface of the environment



def check_cnnoutput(input_size:list, list_modules):
    x = torch.randn(1, *input_size)
    for module in list_modules:
        x = module(x)
    return x.shape[1]


def init_orhtogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)


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
