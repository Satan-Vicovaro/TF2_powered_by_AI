
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from collections import deque,defaultdict
import globals as gl
from sklearn.cluster import KMeans


import TfBot as tf
import pickle
import csv
import os
import copy
import math

#Hyperparameters
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-2 #actor learning rate
CRITIC_LR = 1e-2 #critic learning rate
BUFFER_SIZE = int(1e6) # buffer for ReplayBuffer
BATCH_SIZE = 64

DENSE_DIM = 20

STATE_DIM = 6
ACTION_DIM = 2

CLUSTER_NUM = 10

START_PITCH_CAP = 89

MAX_PITCH_VALUE = 89

class Actor(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, DENSE_DIM ),
            nn.BatchNorm1d(20), 
            nn.ReLU(),
            #nn.Linear(DENSE_DIM, DENSE_DIM),
            #nn.ReLU(),
            nn.Linear(DENSE_DIM, DENSE_DIM),
            nn.BatchNorm1d(20), 
            nn.ReLU()
            )

        # our output as mean 
        self.mean_head = nn.Sequential(
            nn.Linear(DENSE_DIM, action_dim),
            nn.Tanh()  # output in [-1, 1] (is scaled in forward() function)
        )

        # our output as propabiliy
        self.log_std_head = nn.Linear(DENSE_DIM, action_dim)
        self.upper_pitch_cap = START_PITCH_CAP


    def forward(self, state):
        x = self.shared_net(state)

        raw_mean = self.mean_head(x)  # in [-1, 1]

        # scale and offset means
        device = raw_mean.device
        scales = torch.tensor([180.0, self.upper_pitch_cap], device=device)
        
        # shift for [0,400) (the excesive angle does produce error, its properly converted in this case), [-89,89]
        offsets = torch.tensor([1.0, 0.0], device=device) 

        mean = (raw_mean + offsets) * scales

        log_std = self.log_std_head(x)
        std = torch.exp(log_std.clamp(-20, 2))  # numerical stability

        return mean, std
    
    def increase_pitch_cap(self,step):
        self.upper_pitch_cap += step
        if self.upper_pitch_cap > MAX_PITCH_VALUE:
            self.upper_pitch_cap = MAX_PITCH_VALUE

    

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
        return self.net(torch.cat([state, action], dim = 1))
    
class ReplayBuffer: #might not me necessary but AI is forgetfull
    
    def __init__(self, max_size = BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.clustered_samples = {}

    
    def add(self, transition):
        """   
        Appends sample to batch. \n 
        transition: states, actions, rewards \n
        states = [x1, y1, z1, x2, y2, z2] \n 
        actions = [pitch, yaw] \n
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size = BATCH_SIZE):
        """
        Outputs a random sample as FloatTensor in format: \n
        states, actions
        """
        samples = random.sample(self.buffer, batch_size)

        states, actions = zip(*samples)
        #return (states , actions) 
        return (
            torch.stack(states),
            torch.stack(actions),
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

    def load_from_file_csv(self):
        path = "statistics_and_data/training_data.csv"
        with open(path, "r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                t_data = torch.tensor(eval(row["Input_data"]))
                angle = torch.tensor(eval(row["Angle"]))
                self.add((t_data, angle))
            
    def load_from_file_pickle(self):
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

    def split_data_into_sectors(self, num_sectors = 8, center=(0.0, 0.0)):
        """
        Splits data into angular sectors based on angle from a center point (cx, cy)
        """
        states, actions = zip(*self.buffer) 
        data = torch.stack(states)

        x1 = data[:,0]
        y1 = data[:,1]

        cx, cy = center
        
        # Translate to new origin
        dx = x1 - cx
        dy = y1 - cy
        
        # Compute angle from origin (in radians, range [-pi, pi])
        angles = torch.atan2(dy,dx)

        # Normalize angle to [0, 2pi]
        angles = (angles + 2 * math.pi) % (2 * math.pi)

        # Sector size in radians
        sector_size = 2 * math.pi / num_sectors

        # Assign each point to a sector [0, num_sectors - 1]
        sector_ids = (angles / sector_size).floor().long()
  
        # Create buckets
        buckets = [ReplayBuffer() for _ in range(num_sectors)]
        for i, sector in enumerate(sector_ids):
            buckets[sector].add((states[i],actions[i]))

        return buckets
    
    def clusterize(self):
        q_data = torch.tensor([list(q) for _, q in self.buffer])  # shape [N, 2]

        num_clusters = CLUSTER_NUM  # You can tune this

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(q_data)  # shape [N]

        c_samples = defaultdict(list)
        for sample, cluster_id in zip(self.buffer, cluster_ids):
            c_samples[cluster_id].append(sample)
        self.clustered_samples =c_samples


    def sample_from_cluster(self, batch_size = BATCH_SIZE):
        samples = random.sample(self.clustered_samples[0], batch_size)
        states, actions = zip(*samples)
        #return (states , actions) 
        return (
            torch.stack(states),
            torch.stack(actions),
        )
    
# Soft Update
def soft_update(target, source, tau):
    for t_param, param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)


def evaluate(state, action):
    # Example dummy reward: negative L2 norm (encourage action close to 0)
    return -np.linalg.norm(action)


def transition_function(state, action):
    return state + 0.1 * action + np.random.normal(0, 0.01, size=STATE_DIM)


def sample_random_state():critic_optimizer = optim.Adam(critic.parameters(), lr = CRITIC_LR)

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

 
 