import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class DQN(nn.module):
    def __init__(self, n_observations, n_actions, learing_rate):
        super(DQN,self).__init__()
        
        self.layer1 = nn.Linear(n_observations,128) # input
        self.layer2 = nn.Linear(128,128) # one middle layer (idk if its enough)
        self.layer3 = nn.Linear(128,n_actions) # out

        self.optimizer = optim.Adam(self.parameters(),lr= learing_rate) # Who is Adam?
        self.loss = nn.MSELoss() # means square error loss
        self.devide = torch.device(
                "cuda" if torch.cuda.is_available() else # for RTX lovers
                "cpu" 
                )
        
        self.to(self.devide) #load data to device
        

    # propagation function
    def forward(self, state):
        x = F.relu(self.layer1(state)) # relu activation function
        x = F.relu(self.layer2(x))
        action = self.layer3(x)
        return action
    

class Agent():
    # epsilon = how much action we spend on exploring
    # gammma = future action rewards (propably not needed for us)
    # batch_size = in our case number of shooters in tf2
    # epislon_end = minimal value of epsion (almost not exploring)
    
    def __init__(self, gamma, epsilon,learning_rate, input_dims, batch_size, n_actions,max_mem_size = 100000, epsilon_end =0.01, epsilon_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        self.learning_rate =  learning_rate
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)] # in our case 2, pitch an yaw
        self.mem_size = max_mem_size
        self.mem_cur_size = 0
        
        self.Q_eval = DQN(6,2,self.learning_rate) # 6 input paramams (x,y,z) (z,y,z) and 2 output params (pitch, yaw)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float64)

        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float64)
        
        
        self.action_memory = np.zeros(self.mem_size)
        pass