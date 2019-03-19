
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

def conv_output(input_w, filter_w, stride_w):
    return ((input_w - filter_w) / stride_w) +1
def pool_output(input_w, stride_w):
    return floor(input_w / stride_w) 

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=None):
        
        super(QNetwork, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        # fully connected 1
        self.fc1 = nn.Linear(state_size, 256)           # <- input size = state size
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))  # <- using uniform distribution to init the weights of all layers
        
        # fully connected 2
        self.fc2 = nn.Linear(256, 256)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu')) 
        
        # fully connected 3 (output)
        self.fc3 = nn.Linear(256, action_size)           # <- output size = action size
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu')) 

    def forward(self, state): # environment state, otherwise commonly known as x
        """Build a network that maps state -> action values."""
        state   = F.relu(self.fc1(state))
        state   = F.relu(self.fc2(state))
        actions = self.fc3(state)
        return actions

