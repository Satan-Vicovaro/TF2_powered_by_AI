
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque
import globals as gl

import TfBot as tf
import pickle
import csv
import os
import copy

#Hyperparameters
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-2 #actor learning rate
CRITIC_LR = 1e-2 #critic learning rate
BUFFER_SIZE = int(1e6) # buffer for ReplayBuffer
BATCH_SIZE = 64



STATE_DIM = 6
ACTION_DIM = 2

class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # our output as mean 
        self.mean_head = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Tanh()  # output in [-1, 1] (is scaled in forward() function)
        )

        # our output as propabiliy
        self.log_std_head = nn.Linear(32, action_dim)


    def forward(self, state):
        x = self.shared_net(state)

        raw_mean = self.mean_head(x)  # in [-1, 1]
        scales = torch.tensor([179.0, 89.0], device=raw_mean.device)
        mean = raw_mean * scales # now mean[0] in [-179,179] and mean[1] in [-89,89] 

        log_std = self.log_std_head(x)
        std = torch.exp(log_std.clamp(-20, 2)) # clamp for numerical stability
        return mean, std
    

class Critic(nn.Module):
    def __init__(self, state_dim = STATE_DIM, action_dim= ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim,32), # in
            nn.ReLU(),

            nn.Linear(32,32), # middle
            nn.ReLU(),

            nn.Linear(32,1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim = 0))
    
class ReplayBuffer: #might not me necessary but AI is forgetfull
    
    def __init__(self, max_size = BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, transition):
        """   
        Appends sample to batch. \n 
        transition: states, actions, rewards \n
        states = [x1, y1, z1, x2, y2, z2] \n 
        actions = [pitch, yaw] \n
        rewards = float \n
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """
        Outputs a random sample as FloatTensor in format: \n
        states, actions, rewards
        """
        samples = random.sample(self.buffer, batch_size)

        states, actions, rewards, = zip(*samples)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),

        )
    
    def __len__(self):
        return len(self.buffer)
    
    def update_file_csv(self):
        path = "statistics_and_data/training_data.csv"
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            
            if os.path.exists(path) and os.stat(path).st_size == 0:
                writer.writerow(["Input_data", "Angle", "Evaluation"])

            while self.buffer:
                input_data, angle = self.buffer.pop()
                writer.writerow([input_data.tolist(), angle.tolist()])

    
    def update_file_pickle(self):
        with open("statistics_and_data/training_data_pickle", "ab+") as file:
            while self.buffer:
                item = self.buffer.pop()
                pickle.dump(item,file)

    def update_file_pickle_csv(self):
        coppy_buffer = copy.deepcopy(self.buffer)
        self.update_file_csv()
        self.buffer = coppy_buffer
        self.update_file_pickle()        

    def load_form_file_pickle(self):
        with open("statistics_and_data/training_data_pickle", "rb") as file:
            while True:
                try:
                    item = pickle.load(file)
                    self.buffer.append(item)
                except EOFError:
                    break
    
    def update_queue(self,s_bots: dict[np.int64, tf.TfBot], t_bot:tf.TfBot, angles:torch.Tensor):
        """
            Appends to buffer all information information of direct hit shot
        """
        for i , s_bot in enumerate(s_bots.values()):
            if s_bot.damage_dealt != 0:
                self.add((torch.tensor([s_bot.pos_x, s_bot.pos_y, s_bot.pos_z, t_bot.pos_x, t_bot.pos_y, t_bot.pos_z]),angles[i]))
    

    
# Soft Update
def soft_update(target, source, tau):
    for t_param, param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)


def evaluate(state, action):
    # Example dummy reward: negative L2 norm (encourage action close to 0)
    return -np.linalg.norm(action)


def transition_function(state, action):
    return state + 0.1 * action + np.random.normal(0, 0.01, size=STATE_DIM)


def sample_random_state():
    return np.random.uniform(-1, 1, size=6)


def state_from_tfbot(b:tf.TfBot, t:tf.TfBot):
    return torch.Tensor[b.pos_x, b.pos_y, b.pos_z, t.pos_x, t.pos_y, t.pos_z]



# actual AI setup
actor = Actor()
actor_target = Actor()
actor_target.load_state_dict(actor.state_dict()) # coppying our neural network

critic = Critic()
critic_target = Critic()
critic_target.load_state_dict(critic.state_dict()) # coppying our neural network

# Who is Adam? (Adaptive Moment Estimation) btw
# Adam is better than stochastic gradient decent (SGD) (better back propagation)
actor_optimizer = optim.Adam(actor.parameters(), lr = ACTOR_LR) 
critic_optimizer = optim.Adam(critic.parameters(), lr = CRITIC_LR)


replay_buffer = ReplayBuffer()

