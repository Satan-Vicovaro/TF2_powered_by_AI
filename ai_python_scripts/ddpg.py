import numpy as np
import torch
from torch import nn

import random
from collections import deque


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import time
import numpy as np
import csv

class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape),  hidden_dim),
            nn.BatchNorm1d(20), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(hidden_dim, np.prod(action_space.shape)),
            nn.Tanh()
        )
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low)/ 2, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32)
        )
        
    def forward(self, observation):
        #print("Input observation shape:", observation.shape)

        action = self.network(observation)
        return action * self.action_scale + self.action_bias
    


class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape) + np.prod(action_space.shape),  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=1))
    



class OrnsteinUhlenbeckNoise:
    "OU noise generator class, used to add OU noise to actions."
    def __init__(self, size: int, mu=0.0, sigma=0.1, theta=0.15):
        self.mu = mu * torch.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.size = size
        self.reset()
        
    def reset(self):
        "Resets noise to mean."
        self.state = self.mu.clone()
        
    def sample(self):
        "Returns next value generated in process."
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.size)
        self.state += dx
        return self.state.clone()
    
    def sized_sample(self, size):
        "Returns next n values generated in process."
        states = torch.zeros(size,2)
        for i in range(0,size):
            dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.size)
            self.state += dx
            states[i] = self.state.clone()
        return states
    
    




class ReplayBuffer:
    def __init__(self, capacity, num_steps=1, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.num_steps = num_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=num_steps)
        
    def add(self, transition):
        "Pushes transition to buffer and handles n-step logic if required."
        assert len(transition) == 6, "Use new Gym step API: (s, a, r, s', ter, tru)"
        if self.num_steps == 1:
            observation, action, reward, next_observation, terminated, truncated = transition
            self.buffer.append((observation, action, reward, next_observation, terminated))
        else:
            self.n_step_buffer.append(transition)
            
            # Calculate n-step reward
            _, _, _, final_observation, final_termination, final_truncation = transition
            n_step_reward = 0.0
            for _, _, reward, _, _, _ in reversed(self.n_step_buffer):
                  n_step_reward = n_step_reward * self.gamma + reward
            observation, action, _, _, _, _ = self.n_step_buffer[0]

            # If n-step buffer is full, append to main buffer
            if len(self.n_step_buffer) == self.num_steps:
                self.buffer.append((observation, action, n_step_reward, final_observation, final_termination))
            
            # If done, clear n-step buffer
            if final_termination or final_truncation:
                self.n_step_buffer.clear()
                
    def sample(self, batch_size):
        "Samples a batch of experiences for learner to learn from."
        observations, actions, rewards, next_observations, terminations = \
            zip(*random.sample(self.buffer, batch_size))
        return observations, actions, rewards, next_observations, terminations
        
    def __len__(self):
        return len(self.buffer)
    
    def save_to_file_csv(self):
        path = "statistics_and_data/training_data_DDPG.csv"
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            
            if os.path.exists(path) and os.stat(path).st_size == 0:
                writer.writerow(["observations", "actions", "rewards", "next_observations"])

            while self.buffer:
                observations, actions, rewards, next_observations, terminations = self.buffer.pop()
                writer.writerow([observations.tolist(), actions.tolist(),rewards,next_observations])
        

    def load_from_file_csv(self):
        path = "statistics_and_data/training_data_DDPG.csv"
        try:
            with open(path, "r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    observations = torch.tensor(eval(row["observations"]))
                    actions = torch.tensor(eval(row["actions"]))
                    rewards = torch.tensor(eval(row["rewards"]))
                    next_observations = torch.tensor(eval(row["next_observations"]))
                    self.add((observations, actions,rewards, next_observations, False,False))
        except:
            lg.logger.warning("Error when loading from file: training_data_DDPG.csv")
    








import logger as lg
import globals as gl
from queue import Queue
import TfBot as tf
import threading
import squirrel_api as sq

def user_input_listener(player_input_messages:Queue):
        lg.logger.info("listening for player input") 

        while not gl.end_program.is_set():
            try:
                user_input = input("[You] > ").strip()
                if user_input.lower() == "exit":
                    gl.end_program.set()
                elif user_input.lower() == "start":
                    player_input_messages.put("start |")
                    gl.send_message.set()
                    # watiging for squirrel to init it self 
                    time.sleep(0.25) 
                    gl.start_program.set()
                    continue
                elif user_input.lower() == "load nn":
                    lg.logger.info("Neural network will be loaded from file")
                    gl.load_neural_network = True
                    continue
                elif user_input.lower() == "debug on": 
                    lg.enable_debug()
                elif user_input.lower() == "debug off": 
                    lg.disable_debug()
                else:
                    player_input_messages.put(user_input + " |")  

                lg.logger.debug("User_listener: Got input form player")
                gl.send_message.set()          
                                    
            except (KeyboardInterrupt,EOFError):
                lg.logger.error("Keyboard error occured")
                gl.end_program.set()




class CustomActionSpace:
    "Parameters that descibes our inputs and outputs"
    def __init__(self,high:np.ndarray, low:np.ndarray):
        self.low = low
        self.high = high
        self.shape = self.low.shape



class Enviroment:
    "Our tf enviroment is moved here"

    def __init__(self):
        self.bots: dict[np.int64,tf.TfBot] = dict()         # all of our bots
        self.t_bots: dict[np.int64,tf.TfBot] = dict()       # sub category: target bots
        self.s_bots: dict[np.int64,tf.TfBot] = dict()       # sub category: shooter bots
        self.player_input_messages:Queue = Queue()          
        self.restart_count = 0
        self.iteration = 0
        self.tf_listener = threading.Thread(target = sq.tf2_listener_and_sender, args = (self.player_input_messages,self.bots, ), daemon = True)
        self.user_listener = threading.Thread(target = user_input_listener, args = (self.player_input_messages, ), daemon = True)

        self.tf_listener.start()
        self.user_listener.start()
        pass

    def __del__(self):
            os._exit(0)
            self.tf_listener.join()
            self.user_listener.join()

    def get_observation_and_action_spaces(self):
        action_space = CustomActionSpace(high = np.array([360, 0]),low = np.array([0,-70]))
        observation_space = CustomActionSpace(np.array([100, 100, 100, 100, 100, 100]),np.array([-100, -100, -100, -100, -100, -100]))
        return action_space , observation_space

    
        
    def reset(self) -> torch.Tensor:
        """
        Resets our enviroment and gets initial position,
        in our case we just send tf positions
        """

        while True:
            should_restart = self.request_positions()  
            if not should_restart:
                break
        
        #positions are saved in self.bots.dict
        self.t_bots, self.s_bots = self.dispatch_bots_into_shooters_and_targets()

        #normalizing data
        self.normalize_data()

        
        data = self.crate_training_data()


        return data


    def step(self, angles, observations):
        """
            the evaluation of the function,
            returns:
            next_obsevation (next bots positions, it does not change in our case),
            reward,
            terminated (if we succeed in achieving goal),
            truncated (timeout limit in training session)
        """
        # this loop is kinda weird

        # evaluate previous position

        self.send_tensor_angles(angles)

        # wait for damage response,
        time.sleep(1.25)

        while True:
            should_restart = self.request_damage_data()
            if not should_restart:
                break
        
        while True:
            should_restart = self.request_bullet_data()
            if not should_restart:
                break

        rewards = self.evaluate(angles, observations)

        if self.iteration % 1 == 0:
            # next positions
            self.player_input_messages.put("change_target_pos|") 
            gl.send_message.set()
            time.sleep(0.2)

        while True:
            should_restart = self.request_positions()  
            if not should_restart:
                break
        
        # this is better 
        self.nomalize_missiles()
        self.normalize_data()

        #next positions
        self.t_bots,self.s_bots = self.dispatch_bots_into_shooters_and_targets()
        next_observation = self.crate_training_data() #observation does not changes


        self.reset_damage_dealt()
        self.iteration += 1
        lg.logger.info("Iteration: " + str(self.iteration))

        return next_observation, rewards, torch.zeros_like(rewards,dtype=torch.bool), torch.zeros_like(rewards,dtype=torch.bool)

        
    def reset_damage_dealt(self):
        for bot in self.bots.values():
            bot.damage_dealt = 0
        
    def evaluate(self,angles, observations:torch.Tensor):
        

        # proper evaluation function that uses ⭐️real math⭐️ to calculate proper value
        # value is in range [-1, 2] 
        rewards = torch.zeros((angles.shape[0]))

        s_pos, t_pos = observations.split(3,dim=1)
        v_shooter_target = t_pos - s_pos # vector: shooter ---> target
        for i, s_bot in enumerate(self.s_bots.values()):

            missile_pos = torch.tensor([s_bot.m_x, s_bot.m_y, s_bot.m_z])
            v_shooter_missile = missile_pos - s_pos[i] # vector: shooter ---> missile
            
            # angle between shooter ---> missile and shooter ---> target
            cosine_angle = torch.dot(v_shooter_missile, v_shooter_target[i]) / (v_shooter_missile.norm() * v_shooter_target[i].norm())
            rewards[i] = cosine_angle 

            if s_bot.damage_dealt > 0:
                rewards[i] *= 2 #aditional reward for hiting target

        lg.logger.debug(rewards)
        lg.logger.info("Average reward: " + str(rewards.mean()))
        lg.logger.info("Sum of rewards: " + str(rewards.sum()))
        return rewards #(rewards - rewards.mean()) / (rewards.std() + 1e-8) #normalized


    def random_action(self):
        angles = torch.zeros(len(self.s_bots),2)
        for i, bot in enumerate(self.s_bots):
            angles[i][0] = torch.tensor(random.uniform(0,360))
            angles[i][1] = torch.tensor(random.uniform(-70, 0))
        return angles

    def request_bullet_data(self):
        # requesting bullets distances from target_bot
        self.player_input_messages.put("send_distances|")
        gl.send_message.set()

        lg.logger.debug("waiting for bullet data")
        # waiting for damage data
        if not gl.received_bullet_data.wait(gl.MAX_DURATION):
            lg.logger.warning("Received bullet data timeout reached, restarting the loop...")
            self.restart_count += 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True
        gl.received_bullet_data.clear() # don't forget to clear the flag
        return False
    
    def request_damage_data(self):
        #requesting damage data
        self.player_input_messages.put( "send_damage|" )
        gl.send_message.set()

        lg.logger.debug("waiting for damage data")
        if not gl.received_damage_data.wait(gl.MAX_DURATION):
            lg.logger.warning("Received damage data timeout reached, restarting the loop...")
            self.restart_count+= 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True 
        gl.received_damage_data.clear()
        return False

    def request_positions(self): 

        #request data postion data
        self.player_input_messages.put("get_position |") #alway end message_type with "|"
        gl.send_message.set()

        lg.logger.debug("Waiting for positions")
        #waiting for positions
        if not gl.received_positions_data.wait(timeout = gl.MAX_DURATION):
            lg.logger.warning("Received position timeout reached, restarting the loop...")
            self.restart_count += 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True
        gl.received_positions_data.clear() #removing flag
        return False
    

    def send_tensor_angles(self, angles:torch.Tensor):
        
        lg.logger.debug("Sending angles")
        
        message = "angles |" 
        
        # message format:
        # bot_id pitch (y) yaw (x)
        for i,bot_id in enumerate(self.s_bots.keys()):
            message += (" {0} {1} {2}\n".format(bot_id, angles[i][0], angles[i][1]))

        self.player_input_messages.put(message)
        gl.send_message.set()



    def send_angles(self, bots: dict[np.int64,tf.TfBot], player_input_messages: Queue):

        message = "angles |"

        # message format:
        # bot_id pitch (y) yaw (x)
        for bot_id,bot in zip (bots.keys(), bots.values()):
            message += (" {0} {1} {2}\n".format(bot_id, bot.pitch, bot.yaw))

        player_input_messages.put(message)
        gl.send_message.set()


    def normalize_data(self):
        for bot in self.bots.values():
            bot.normalize()

    def nomalize_missiles(self):
        for bot in self.bots.values():
            bot.normalize_missiles()

    def dispatch_bots_into_shooters_and_targets(self):

        target_bots: dict[np.int64,tf.TfBot] = {}
        shooter_bots: dict[np.int64,tf.TfBot] = {}

        # seperating shooters from target
        for key, bot in self.bots.items():
            if bot.bot_type == "s":
                shooter_bots[key] = bot
                continue
            if bot.bot_type == "t":
                target_bots[key] = bot        
                continue

            lg.logger.warning("There is a bot without BotType?\n",  bot)
        return target_bots,shooter_bots
    
    def crate_training_data(self): 
        """
            returns:
            torch.tensor(s_x, s_y, s_z, t_x, t_y, t_z)
        """
        if len(self.t_bots) > 1:
            lg.logger.warning("We have more than one target bot")

        t_bot:tf.TfBot = next(iter(self.t_bots.values()))

        return torch.tensor([(bot.pos_x, bot.pos_y, bot.pos_z, t_bot.pos_x, t_bot.pos_y,t_bot.pos_z) for bot in self.s_bots.values()], dtype=torch.float32)
    

class DDPGConfig:
    env_name: str             = 'TF2-missile-learner'  # Environment name
    agent_name: str           =      'DDPG'  # Agent name
    device: str               =       'cpu'  # Torch device
    checkpoint: bool          =       True  # Periodically save model weights
    num_checkpoints: int      =          20  # Number of checkpoints/printing logs to create
    verbose: bool             =        True  # Verbose printing
    total_steps: int          =     100_000  # Total training steps
    target_reward: int | None =        2000  # Target reward used for early stopping
    learning_starts: int      =         100  # Begin learning after this many steps
    gamma: float              =        0.99  # Discount factor
    lr: float                 =        0.e-4 # Learning rate
    hidden_dim: int           =         20   # Actor and critic network hidden dim
    buffer_capacity: int      =     100_000  # Maximum replay buffer capacity
    batch_size: int           =           20 # Batch size used by learner
    num_steps: int            =           1  # Number of steps to unroll Bellman equation by
    tau: float                =       0.005  # Soft target network update interpolation coefficient
    grad_norm_clip: float     =        40.0  # Global gradient clipping value
    noise_sigma: float        =         0.05 # OU noise standard deviation
    noise_theta: float        =        0.01  # OU noise reversion rate    





class Logger:
    "Used to track episode lengths, returns, and total steps."
    def __init__(self, total_steps: int, num_checkpoints: int):
        self.current_step = 0
        self.current_episode = 1
        self.current_return = 0.0
        self.current_length = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.custom_logs = {}
        self.custom_log_keys = []
        self.start_time = time.time()
        self.total_steps = total_steps
        self.num_checkpoints = num_checkpoints
        self.checkpoint_interval = max(1, self.total_steps // self.num_checkpoints)
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_step = 0
        self.header_printed = False
        
        # Logger settings
        self.log_interval = 1    # Print logs every log_interval timesteps
        self.window = 20         # Use this many items from recent logs
    
    def log(self, reward: float, termination: bool, truncation: bool, **kwargs):
        "Updates logger with latest rewards, done flags and any custom logs."
        self.current_step += 1
        self.current_return += reward
        self.current_length += 1

        # Update tracked statistics
        if termination or truncation:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_episode += 1
            self.current_return = 0.0
            self.current_length = 0

        # Update custom_logs with any additional keyword arguments
        for key, value in kwargs.items():
            if key not in self.custom_log_keys:
                self.custom_log_keys.append(key)
            self.custom_logs[key] = value

    def print_logs(self):
        "Prints training progress with headers and updates."
        if self.current_step % self.log_interval == 0 and len(self.episode_returns) > 0:
            elapsed_time = time.time() - self.start_time
            
            # FPS based on last checkpoint
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            time_since_checkpoint = time.time() - self.last_checkpoint_time
            fps = steps_since_checkpoint / time_since_checkpoint if time_since_checkpoint > 0 else 0

            # Calculate other metrics
            progress = 100 * self.current_step / self.total_steps
            mean_reward = np.mean(
                self.episode_returns[-self.window:]
            ) if len(self.episode_returns) >= self.window else np.mean(self.episode_returns)
            mean_ep_length = np.mean(
                self.episode_lengths[-self.window:]
            ) if len(self.episode_lengths) >= self.window else np.mean(self.episode_lengths)
            
            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

            if not self.header_printed:
                log_header = (
                    f"{'Progress':>8}  |  "
                    f"{'Step':>8}  |  "
                    f"{'Episode':>8}  |  "
                    f"{'Mean Rew':>8}  |  "
                    f"{'Mean Len':<7}  |  "
                    f"{'FPS':>6}  |  "
                    f"{'Time':>8}"
                )
                # Append custom log headers
                for key in self.custom_log_keys:
                    log_header += f"  |  {key:>{len(key)}}"
                print(log_header)
                self.header_printed = True

            log_string = (
                f"{progress:>7.1f}%  |  "
                f"{self.current_step:>8,}  |  "
                f"{self.current_episode:>8,}  |  "
                f"{mean_reward:>8.2f}  |  "
                f"{mean_ep_length:>8.1f}  |  "
                f"{fps:>6,.0f}  |  "
                f"{formatted_time:>8}"
            )
            # Append custom log values
            for key in self.custom_log_keys:
                value = self.custom_logs.get(key, 0)
                # Format based on the type of value
                if isinstance(value, float):
                    log_string += f"  |  {value:>{len(key)}.2f}"
                elif isinstance(value, int):
                    log_string += f"  |  {value:>{len(key)}d}"
                else:
                    log_string += f"  |  {str(value):>{len(key)}}"
            print(f"\r{log_string}", end='')
        
        # Check if a checkpoint is reached
        if self.current_step % self.checkpoint_interval == 0:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

    @property
    def logs(self):
        return  {
            'total_steps': self.current_step,
            'total_episodes': self.current_episode - 1,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'best_reward': np.max(self.episode_returns) if len(self.episode_returns) > 0 else None,
            'total_duration': time.time() - self.start_time,
            'mean_fps': self.current_step / (time.time() - self.start_time + 1e-6),
            'custom_logs': self.custom_logs
        }


class DDPG:
    def __init__(self, env:Enviroment):
        config = DDPGConfig()
        self.device = config.device
        

        self.env = env
        action_space, observation_space= self.env.get_observation_and_action_spaces()
        
        self.actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.soft_update(self.actor, self.target_actor, 1.0)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        
        self.critic = CriticNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.target_critic = CriticNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.soft_update(self.critic, self.target_critic, 1.0)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        
        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_steps, config.gamma)
        self.noise_generator = OrnsteinUhlenbeckNoise(
            size=np.prod(action_space.shape), mu=0.0, sigma=config.noise_sigma, theta=config.noise_theta
        )
        self.config = config

    def update_file_DDPG(self, observations:torch.Tensor, actions, rewards:torch.Tensor, next_observations, terminated):
        path = "statistics_and_data/training_data_DDPG.csv"
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            
            if os.path.exists(path) and os.stat(path).st_size == 0:
                writer.writerow(["observations", "actions", "rewards", "next_observations"])
            else:
                writer.writerow([observations.tolist(), actions.tolist(), float(rewards) ,next_observations.tolist()])
    

    def checkpoint(self, steps):
        "Saves model weights to disk."
        if not os.path.exists('models'):
            os.makedirs('models')
        checkpoint_path = f"models/{self.config.agent_name}_{self.config.env_name}_{steps}.pth"
        torch.save(self.actor.state_dict(), checkpoint_path)


    def soft_update(self, online, target, tau):
        "Performs a soft update of the target network parameters."
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


    def select_action(self, observation, add_noise=False):
        "Selects an action using the current policy with optional noise."
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(observation_tensor).squeeze(0)
            if add_noise:
                noise = self.noise_generator.sized_sample(len(action)).to(self.device)
                noise *= self.actor.action_scale
                action = torch.clamp(action + noise, min=-self.actor.action_scale*2, max=self.actor.action_scale*2)
            return action.cpu().numpy()


    def learn(self):
        "Perform a single learning step."
        # Sample and format experience data
        observations, actions, rewards, next_observations, terminations =\
            self.buffer.sample(self.config.batch_size)
        
        observations      = torch.tensor(np.array(observations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        actions           = torch.tensor(np.array(actions), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        rewards           = torch.tensor(np.array(rewards), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size,  1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        terminations      = torch.tensor(np.array(terminations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size,  1)

        #print(observations.shape)
        #print(next_observations.shape)
        # Critic loss and param update
        # Target computation using n-step Bellman equation
        with torch.no_grad():
            next_state_q = self.target_critic(next_observations, self.target_actor(next_observations))
            target_q = rewards + self.config.gamma ** self.config.num_steps * (1.0 - terminations) * next_state_q  

        # Forward pass with critic network to get predicted Q-value for current state 
        current_action_q = self.critic(observations, actions)

        # Critic loss defined as mean squared temporal difference error 
        critic_loss = F.mse_loss(current_action_q, target_q)

        # Backward pass and optimiser step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        # Actor loss and param update
        # Forward pass with the critic from current state
        current_action_q = self.critic(observations, self.actor(observations))
        
        # Policy gradient loss for actor, adjust action in the direction that increases its Q-value
        actor_loss = -(current_action_q).mean()

        # Backward pass and optimiser step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.actor, self.target_actor, self.config.tau)
        self.soft_update(self.critic, self.target_critic, self.config.tau)

    
    def train(self):
        "Trains DDPG agent based on the provided configuration."
        if self.config.verbose:
            print(f"Training {self.config.agent_name} agent...\n")
        
        # Initialise Logger
        logger = Logger(total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints)
        
        self.buffer.load_from_file_csv()

        # Reset environment
        observations = self.env.reset()
        
        # Main training loop
        for step in range(1, self.config.total_steps + 1):
            
            # Select action
            if step > self.config.learning_starts:
                actions = self.select_action(observations, add_noise=True)
            else:
                # Random if not yet learning
                actions = self.env.random_action()
                
            # Environment step
            next_observations, rewards, terminated, truncated = self.env.step(actions,observations)
            
            # Update logs
            #logger.log(reward, terminated, truncated)
            for i in range(0,len(next_observations)):
                if step < self.config.learning_starts:
                    self.update_file_DDPG(observations[i], actions[i], rewards[i], next_observations[i], terminated[i])
                
                # Push experience to buffer
                self.buffer.add((observations[i], actions[i], rewards[i], next_observations[i], terminated[i], truncated[i]))

                
            
            for _ in range(0,30):
                # Perform learning step
                if len(self.buffer) > self.config.batch_size and step >= self.config.learning_starts:
                    self.learn()

            # Reset environment and noise if episode ended
            if terminated.any() or truncated.any():
                next_observations, _ = self.env.reset()
                self.noise_generator.reset()
            observations = next_observations
                
            # Print training info if verbose
            if self.config.verbose:
                logger.print_logs()
                
            # Save weights if checkpointing
            if self.config.checkpoint and step % logger.checkpoint_interval == 0:
                self.checkpoint(step)
                
            # Check stopping condition
            if self.config.target_reward is not None and len(logger.episode_returns) >= 20:
                mean_reward = np.mean(logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved. Training stopped.")
                    #break

        # Training ended
        if self.config.verbose:
            print("\nTraining complete.")
        
        return logger.logs
    


def main():
    tf2_env = Enviroment()

    #start program
    gl.start_program.wait()
    
    ddbg = DDPG(tf2_env)
    ddbg.train()

if __name__ == "__main__":
    main()