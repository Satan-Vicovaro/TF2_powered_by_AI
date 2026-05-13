import numpy as np
import torch
from torch import nn

import random
from collections import deque
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import time
import numpy as np
import csv

from user_listener import UserListener
from data_collector import DataCollector, shared_collector, Severity
from dummy_environment import DummyEnvironment
from environment import Environment


class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, np.prod(action_space.shape)),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32),
        )

    def forward(self, observation):
        return self.network(observation)


class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape) + np.prod(action_space.shape), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        states = torch.zeros(size, 2)
        for i in range(0, size):
            dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.size)
            self.state += dx
            states[i] = self.state.clone()
        return states


import pickle


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
                self.buffer.append(
                    (observation, action, n_step_reward, final_observation, final_termination)
                )

            # If done, clear n-step buffer
            if final_termination or final_truncation:
                self.n_step_buffer.clear()

    def sample(self, batch_size):
        "Samples a batch of experiences for learner to learn from."
        observations, actions, rewards, next_observations, terminations = zip(
            *random.sample(self.buffer, batch_size)
        )
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
                writer.writerow(
                    [observations.tolist(), actions.tolist(), rewards, next_observations]
                )

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
                    self.add((observations, actions, rewards, next_observations, False, False))
        except:
            lg.logger.warning("Error when loading from file: training_data_DDPG.csv")

    def dump_buffer(self, path):
        try:
            with open("statistics_and_data/" + path + "/queue_dump.pkl", "wb") as f:
                pickle.dump(self.buffer, f)
        except:
            lg.logger.warning("Could not dump buffer")
            return
        finally:
            lg.logger.info("Buffer dumped succesfully")

    def load_dumped_buffer(self, path):
        try:
            with open("statistics_and_data" + path + "/queue_dump.pkl", "rb") as f:
                buffer_pickle = pickle.load(f)

                for item in buffer_pickle:
                    self.buffer.append(item)

        except:
            lg.logger.warning("Could not load dumped buffer")


import logger as lg
import globals as gl
from queue import Queue
import TfBot as tf
import threading
import squirrel_api as sq


class DDPGConfig:
    env_name: str = "TF2-missile-learner"  # Environment name
    agent_name: str = "DDPG"  # Agent name
    device: str = "cpu"  # Torch device
    checkpoint: bool = True  # Periodically save model weights
    num_checkpoints: int = 10  # Number of checkpoints/printing logs to create
    verbose: bool = False  # Verbose printing
    total_steps: int = 50_000  # Total training steps
    target_reward: int | None = 2  # Target reward used for early stopping
    learning_starts: int = 100  # Begin learning after this many step
    gamma: float = 0.99  # Discount factor
    lr: float = 0.001  # Learning rate
    hidden_dim: int = 64 * 4  # Actor and critic network hidden dim
    buffer_capacity: int = 10_000  # Maximum replay buffer capacity
    batch_size: int = 32 * 2  # Batch size used by learner
    num_steps: int = 1  # Number of steps to unroll Bellman equation by
    tau: float = 0.005  # Soft target network update interpolation coefficient
    grad_norm_clip: float = 1000.0  # Global gradient clipping value

    noise_sigma: float = 0.20  # OU noise standard deviation
    sigma_decrease_coef: float = 0.02
    min_noise_sigma: float = 0.00

    noise_sigma_decrease_iteration: int = 500

    noise_theta: float = 0.05  # OU noise reversion rate
    min_noise_theta: float = 0.01
    theta_decrease_coef: float = 0.001


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
        # self.checkpoint_interval = 10
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_step = 0
        self.header_printed = False

        # Logger settings
        self.log_interval = 100  # Print logs every log_interval timesteps
        self.window = 100  # Use this many items from recent logs

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
            mean_reward = (
                np.mean(self.episode_returns[-self.window :])
                if len(self.episode_returns) >= self.window
                else np.mean(self.episode_returns)
            )
            mean_ep_length = (
                np.mean(self.episode_lengths[-self.window :])
                if len(self.episode_lengths) >= self.window
                else np.mean(self.episode_lengths)
            )

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
            print(f"\r{log_string}", end="")

        # Check if a checkpoint is reached
        if self.current_step % self.checkpoint_interval == 0:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

    @property
    def logs(self):
        return {
            "total_steps": self.current_step,
            "total_episodes": self.current_episode - 1,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "best_reward": np.max(self.episode_returns) if len(self.episode_returns) > 0 else None,
            "total_duration": time.time() - self.start_time,
            "mean_fps": self.current_step / (time.time() - self.start_time + 1e-6),
            "custom_logs": self.custom_logs,
        }


class DDPG:
    def __init__(self, env):
        config = DDPGConfig()
        self.device = config.device

        self.iteration = 1
        self.pitch_angle_cap = -5
        self.pitch_max = -70

        self.env = env
        action_space, observation_space = self.env.get_observation_and_action_spaces()

        self.actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(
            self.device
        )

        self.target_actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(
            self.device
        )
        # self.soft_update(self.actor, self.target_actor, 1.0)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)

        self.critic = CriticNetwork(observation_space, action_space, config.hidden_dim).to(
            self.device
        )
        self.target_critic = CriticNetwork(observation_space, action_space, config.hidden_dim).to(
            self.device
        )
        # self.soft_update(self.critic, self.target_critic, 1.0)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_steps, config.gamma)
        self.noise_generator = OrnsteinUhlenbeckNoise(
            size=np.prod(action_space.shape),
            mu=0.0,
            sigma=config.noise_sigma,
            theta=config.noise_theta,
        )
        self.config = config
        self.is_loaded = False

        if gl.load_neural_network:
            self.is_loaded = True
            checkpoint_data = torch.load("statistics_and_data/smart_1.pth")
            self.actor.load_state_dict(checkpoint_data["actor"])
            self.critic.load_state_dict(checkpoint_data["critic"])

            self.soft_update(self.actor, self.target_actor, 1.0)
            self.soft_update(self.critic, self.target_critic, 1.0)

            if "actor_optimizer" in checkpoint_data:
                self.actor_optimizer.load_state_dict(checkpoint_data["actor_optimizer"])
                self.critic_optimizer.load_state_dict(checkpoint_data["critic_optimizer"])

            self.noise_generator.sigma = self.config.min_noise_sigma

    def update_file_DDPG(
        self,
        observations: torch.Tensor,
        actions,
        rewards: torch.Tensor,
        next_observations,
        terminated,
    ):
        path = "statistics_and_data/training_data_DDPG.csv"
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)

            if os.path.exists(path) and os.stat(path).st_size == 0:
                writer.writerow(["observations", "actions", "rewards", "next_observations"])
            else:
                writer.writerow(
                    [
                        observations.tolist(),
                        actions.tolist(),
                        float(rewards),
                        next_observations.tolist(),
                    ]
                )

    def checkpoint(self, steps):
        "Saves model weights to disk."
        if not os.path.exists("models"):
            os.makedirs("models")
        checkpoint_path = f"models/{self.config.agent_name}_{self.config.env_name}_{steps}.pth"

        checkpoint_data = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)

    def soft_update(self, online, target, tau):
        "Performs a soft update of the target network parameters."
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, observation, add_noise=False):
        "Selects an action using the current policy with optional noise."
        with torch.no_grad():
            # print("Observation: ", observation.shape)
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            action = self.actor(observation_tensor).squeeze(0)
            if add_noise:
                noise = self.noise_generator.sized_sample(len(action)).to(self.device)
                # noise *= self.actor.action_scale
                action = torch.clamp(
                    action + noise,
                    min=-1,
                    max=1,
                )

                # action = torch.clamp(action,min=torch.tensor([0, self.pitch_angle_cap]),max=torch.tensor([720,10]))

                # if self.iteration % 1000 == 0:
                #     self.pitch_angle_cap = max(self.pitch_max,self.pitch_angle_cap - 3)

                if self.iteration % self.config.noise_sigma_decrease_iteration == 0:
                    self.noise_generator.sigma = max(
                        self.config.min_noise_sigma,
                        self.noise_generator.sigma - self.config.sigma_decrease_coef,
                    )

            return action.cpu()

    def learn(self):
        "Perform a single learning step."
        # Sample and format experience data
        observations, actions, rewards, next_observations, terminations = self.buffer.sample(
            self.config.batch_size
        )

        observations = torch.tensor(
            np.array(observations), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, -1)

        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).view(
            self.config.batch_size, -1
        )

        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).view(
            self.config.batch_size, 1
        )

        next_observations = torch.tensor(
            np.array(next_observations), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, -1)

        terminations = torch.tensor(
            np.array(terminations), dtype=torch.float32, device=self.device
        ).view(self.config.batch_size, 1)

        # print(observations.shape)
        # print(next_observations.shape)
        # Critic loss and param update
        # Target computation using n-step Bellman equation
        with torch.no_grad():
            next_state_q = self.target_critic(
                next_observations, self.target_actor(next_observations)
            )
            target_q = (
                rewards
                + self.config.gamma**self.config.num_steps * (1.0 - terminations) * next_state_q
            )

        # Forward pass with critic network to get predicted Q-value for current state
        current_action_q = self.critic(observations, actions)

        # Critic loss defined as mean squared temporal difference error
        critic_loss = F.mse_loss(current_action_q, target_q)
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
        "Tra1ns DDPG agent based on the provided configuration."
        if self.config.verbose:
            print(f"Training {self.config.agent_name} agent...\n")

        # Initialise Logger
        logger = Logger(
            total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints
        )

        # self.buffer.load_from_file_csv()
        #
        # self.buffer.load_dumped_buffer("/18:24")

        # Reset environment
        observations = self.env.reset()

        # Main training loop
        for step in range(1, self.config.total_steps + 1):
            shared_collector.next_iteration()
            # Select action
            if self.is_loaded or step > self.config.learning_starts:
                actions = self.select_action(observations, add_noise=True)
            else:
                # Random if not yet learning
                actions = self.env.random_action()

            # Environment step
            next_observations, rewards, terminated, truncated = self.env.step(
                actions, observations, self.iteration
            )

            # Update logs
            # logger.log(reward, terminated, truncated)
            for i in range(0, len(next_observations)):
                if step < self.config.learning_starts:
                    self.update_file_DDPG(
                        observations[i], actions[i], rewards[i], next_observations[i], terminated[i]
                    )

                # Push experience to buffer
                self.buffer.add(
                    (
                        observations[i],
                        actions[i],
                        rewards[i],
                        next_observations[i],
                        terminated[i],
                        truncated[i],
                    )
                )

            # for _ in range(0, 10):
            # Perform learning step

            # If a model was loaded, require a much larger buffer before starting gradient descent
            # to prevent catastrophic forgetting on a tiny dataset of recent experiences.
            required_buffer_size = 2000 if self.is_loaded else self.config.batch_size
            if len(self.buffer) > required_buffer_size and (
                self.is_loaded or step >= self.config.learning_starts
            ):
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
                shared_collector.save_data()
                # self.env.checkpoint_save_logs()
                # self.buffer.dump_buffer(self.env.logger_dir_name)

            # Check stopping condition
            if self.config.target_reward is not None and len(logger.episode_returns) >= 20:
                mean_reward = np.mean(logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved!")
                    # break

            self.iteration += 1

            if self.iteration % 1000 == 0:
                shared_collector.print_interation_data()
                lg.logger.info("Sigma: {0:.3f}".format(self.env.adaptive_sigma.sigma))

        # Training ended
        if self.config.verbose:
            print("\nTraining complete.")
        shared_collector.save_data()

        return logger.logs

    def act(self):
        logger = Logger(
            total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints
        )

        observations = self.env.reset()

        for step in range(1, self.config.total_steps + 1):
            actions = self.select_action(observations, add_noise=False)

            next_observations, rewards, terminated, truncated = self.env.step(
                actions, observations, self.iteration
            )

            if terminated.any() or truncated.any():
                next_observations, _ = self.env.reset()
                self.noise_generator.reset()
            observations = next_observations

            # Print training info if verbose
            if self.config.verbose:
                logger.print_logs()

            self.iteration += 1


def main():
    user_listener = UserListener()
    user_listener.start()

    # start program
    gl.start_program.wait()

    enviroment = None
    if gl.enviroment_type == "dummy":
        enviroment = DummyEnvironment()
    elif gl.enviroment_type == "normal":
        enviroment = Environment()
    else:
        enviroment = Environment()

    ddbg = DDPG(enviroment)
    if gl.is_learning:
        ddbg.train()
    else:
        ddbg.act()

    user_listener.stop()


if __name__ == "__main__":
    main()
