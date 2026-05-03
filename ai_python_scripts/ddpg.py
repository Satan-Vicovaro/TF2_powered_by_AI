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
from dummy_enviroment import Enviroment as DummyEnviroment


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


class CustomActionSpace:
    "Parameters that descibes our inputs and outputs"

    def __init__(self, high: np.ndarray, low: np.ndarray):
        self.low = low
        self.high = high
        self.shape = self.low.shape


class Enviroment:
    "Our tf enviroment is moved here"

    def __init__(self):
        self.bots: dict[np.int64, tf.TfBot] = dict()  # all of our bots
        self.t_bots: dict[np.int64, tf.TfBot] = dict()  # sub category: target bots
        self.s_bots: dict[np.int64, tf.TfBot] = dict()  # sub category: shooter bots
        self.restart_count = 0
        self.iteration = 0
        self.accuracy_logger = []
        self.avg_reward_logger = []
        self.sum_reward_logger = []
        self.logger_dir_name = datetime.now().strftime("%H:%M")

        self.tf_listener = threading.Thread(
            target=sq.tf2_listener_and_sender,
            args=(
                gl.player_input_messages,
                self.bots,
            ),
            daemon=True,
        )
        gl.end_program.clear()
        self.tf_listener.start()

    def __del__(self):
        gl.end_program.set()

    def get_observation_and_action_spaces(self):
        action_space = CustomActionSpace(high=np.array([360, 89]), low=np.array([0, -89]))
        observation_space = CustomActionSpace(
            np.array([1, 1, 1, 1, 1, 1]), np.array([-1, -1, -1, -1, -1, -1])
        )
        return action_space, observation_space

    def reset(self) -> torch.Tensor:
        """
        Resets our enviroment and gets initial position,
        in our case we just send tf positions
        """

        while True:
            should_restart = self.request_positions()
            if not should_restart:
                break

        # positions are saved in self.bots.dict
        self.t_bots, self.s_bots = self.dispatch_bots_into_shooters_and_targets()

        # normalizing data
        self.normalize_data()

        data = self.crate_training_data()

        return data

    def step(self, angles, observations, iteration):
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

        multipliers = torch.tensor([180.0, 90.0])
        shifts = torch.tensor([180.0, 0.0])
        real_angles = (angles * multipliers) + shifts

        self.send_tensor_angles(real_angles)
        # wait for damage response,
        time.sleep(1.50)

        while True:
            should_restart = self.request_damage_data()
            if not should_restart:
                break

        while True:
            should_restart = self.request_bullet_data()
            if not should_restart:
                break

        self.nomalize_missiles()

        rewards = self.evaluate(angles, observations)

        if iteration % 1 == 0:
            self.request_change_target_position()
            # self.request_change_shooter_positions()

        while True:
            should_restart = self.request_positions()
            if not should_restart:
                break
        self.normalize_data()

        # next positions
        self.t_bots, self.s_bots = self.dispatch_bots_into_shooters_and_targets()
        next_observation = self.crate_training_data()  # observation does not changes

        self.reset_damage_dealt()

        return (
            next_observation,
            rewards,
            torch.zeros_like(rewards, dtype=torch.bool),
            torch.zeros_like(rewards, dtype=torch.bool),
        )

    def reset_damage_dealt(self):
        for bot in self.bots.values():
            bot.damage_dealt = 0

    def evaluate(self, angles, observations: torch.Tensor):
        rewards = torch.zeros((angles.shape[0]))

        for i, s_bot in enumerate(self.s_bots.values()):
            miss_dist = torch.norm(torch.tensor([s_bot.m_miss_x, s_bot.m_miss_y, s_bot.m_miss_z]))
            hit = s_bot.damage_dealt > 0

            if hit:
                rewards[i] = 1.5
            else:
                # sigma tunable: 0.3 ≈ 300 units, adjust to target hitbox size
                sigma = 0.5  # maybe should be lower
                rewards[i] = torch.exp(torch.tensor(-(miss_dist**2) / sigma**2))

                pitch = angles[i, 1]

                # Penalise extreme pitch — angles[:, 1] is in [-1, 1]
                # abs(pitch) near 1.0 means straight up or straight down
                if not (-70 < pitch * 90 < 70):
                    pitch_normalized = pitch.abs()  # [0, 1]
                    pitch_penalty = (
                        pitch_normalized**2
                    )  # soft, quadratic — only bites near extremes
                    rewards[i] -= pitch_penalty

        self.show_and_update_logs(rewards)
        return rewards

    def random_action(self):
        angles = torch.zeros(len(self.s_bots), 2)
        for i, _ in enumerate(self.s_bots):
            angles[i][0] = torch.tensor(random.uniform(-1, 1))
            angles[i][1] = torch.tensor(random.uniform(-1, 1))
        return angles

    def request_bullet_data(self):
        # requesting bullets distances from target_bot
        gl.player_input_messages.put("send_distances|")
        gl.send_message.set()

        lg.logger.debug("waiting for bullet data")
        # waiting for damage data
        if not gl.received_bullet_data.wait(gl.MAX_DURATION):
            lg.logger.warning("Received bullet data timeout reached, restarting the loop...")
            self.restart_count += 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True
        gl.received_bullet_data.clear()  # don't forget to clear the flag
        return False

    def request_damage_data(self):
        # requesting damage data
        gl.player_input_messages.put("send_damage|")
        gl.send_message.set()

        lg.logger.debug("waiting for damage data")
        if not gl.received_damage_data.wait(gl.MAX_DURATION):
            lg.logger.warning("Received damage data timeout reached, restarting the loop...")
            self.restart_count += 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True
        gl.received_damage_data.clear()
        return False

    def request_positions(self):

        # request data postion data
        gl.player_input_messages.put("get_position |")  # alway end message_type with "|"
        gl.send_message.set()

        lg.logger.debug("Waiting for positions")
        # waiting for positions
        if not gl.received_positions_data.wait(timeout=gl.MAX_DURATION):
            lg.logger.warning("Received position timeout reached, restarting the loop...")
            self.restart_count += 1
            lg.logger.warning("Restart count: " + str(self.restart_count))
            return True
        gl.received_positions_data.clear()  # removing flag
        return False

    def send_tensor_angles(self, angles: torch.Tensor):

        lg.logger.debug("Sending angles")

        message = "angles |"

        # message format:
        # bot_id pitch (y) yaw (x)
        for i, bot_id in enumerate(self.s_bots.keys()):
            message += " {0} {1} {2}\n".format(bot_id, angles[i][0], angles[i][1])

        gl.player_input_messages.put(message)
        gl.send_message.set()

    def send_angles(self, bots: dict[np.int64, tf.TfBot], player_input_messages: Queue):

        message = "angles |"

        # message format:
        # bot_id pitch (y) yaw (x)
        for bot_id, bot in zip(bots.keys(), bots.values()):
            message += " {0} {1} {2}\n".format(bot_id, bot.pitch, bot.yaw)

        player_input_messages.put(message)
        gl.send_message.set()

    def normalize_data(self):
        for bot in self.bots.values():
            bot.normalize()

    def nomalize_missiles(self):
        for bot in self.bots.values():
            bot.normalize_missiles()

    def dispatch_bots_into_shooters_and_targets(self):

        target_bots: dict[np.int64, tf.TfBot] = {}
        shooter_bots: dict[np.int64, tf.TfBot] = {}

        # seperating shooters from target
        for key, bot in self.bots.items():
            if bot.bot_type == "s":
                shooter_bots[key] = bot
                continue
            if bot.bot_type == "t":
                target_bots[key] = bot
                continue

            lg.logger.warning("There is a bot without BotType?\n", bot)
        return target_bots, shooter_bots

    def crate_training_data(self):
        """
        returns:
        torch.tensor(s_x, s_y, s_z, t_x, t_y, t_z)
        """
        if len(self.t_bots) > 1:
            lg.logger.warning("We have more than one target bot")

        t_bot: tf.TfBot = next(iter(self.t_bots.values()))

        return torch.tensor(
            [
                (bot.pos_x, bot.pos_y, bot.pos_z, t_bot.pos_x, t_bot.pos_y, t_bot.pos_z)
                for bot in self.s_bots.values()
            ],
            dtype=torch.float32,
        )

    def request_change_target_position(self):
        gl.player_input_messages.put("change_target_pos|")
        gl.send_message.set()
        time.sleep(0.2)

    def request_change_shooter_positions(self):
        center_x = random.uniform(-500, 500)
        center_y = random.uniform(-500, 500)
        center_z = 140.0  # keep them on the ground plane
        radius = random.uniform(50, 400)

        gl.player_input_messages.put(
            f"change_shooter_pos|{center_x:.1f} {center_y:.1f} {center_z:.1f} {radius:.1f}"
        )
        gl.send_message.set()
        time.sleep(0.2)

    def show_and_update_logs(self, rewards: torch.Tensor):

        lg.logger.debug(rewards)

        lg.logger.info("Average reward: {0:.2f}".format(rewards.mean()))
        self.avg_reward_logger.append("{0:.2f}".format(rewards.mean()))
        shared_collector.append("Average_reward", "{0:.2f}".format(rewards.mean()))

        lg.logger.info("Sum of rewards: {0:.2f}".format(rewards.sum()))
        self.sum_reward_logger.append("{0:.2f}".format(rewards.sum()))
        shared_collector.append("Sum_reward", "{0:.2f}".format(rewards.mean()))

        hit_counter = 0
        for s_bot in self.s_bots.values():
            if s_bot.damage_dealt > 0:
                hit_counter += 1

        lg.logger.info("Accuracy: {0:.2f}".format(hit_counter / len(self.s_bots)))
        shared_collector.append("Hit_counter", "{0:.2f}".format(hit_counter / len(self.s_bots)))
        self.accuracy_logger.append("{0:.2f}".format(hit_counter / len(self.s_bots)))

    def checkpoint_save_logs(self):
        Enviroment.create_dir(self.logger_dir_name)

        Enviroment.save_data_to_file(
            str(self.accuracy_logger), "statistics_and_data/" + self.logger_dir_name + "/accuracy"
        )
        Enviroment.save_data_to_file(
            str(self.avg_reward_logger),
            "statistics_and_data/" + self.logger_dir_name + "/avg_reward",
        )
        Enviroment.save_data_to_file(
            str(self.sum_reward_logger),
            "statistics_and_data/" + self.logger_dir_name + "/sum_reward",
        )

    @staticmethod
    def get_next_filename(base_name="file", extension=".txt"):
        index = 1
        while True:
            filename = f"{base_name}_{index}{extension}"
            if not os.path.exists(filename):
                return filename
            index += 1

    @staticmethod
    def save_data_to_file(data, file_name):
        with open(file_name, "w") as file:
            file.write(data)
        lg.logger.info(f"Data saved to {file_name}")

    @staticmethod
    def create_dir(name):
        os.makedirs("statistics_and_data/" + name, exist_ok=True)


class DDPGConfig:
    env_name: str = "TF2-missile-learner"  # Environment name
    agent_name: str = "DDPG"  # Agent name
    device: str = "cpu"  # Torch device
    checkpoint: bool = True  # Periodically save model weights
    num_checkpoints: int = 40  # Number of checkpoints/printing logs to create
    verbose: bool = False  # Verbose printing
    total_steps: int = 30_000  # Total training steps
    target_reward: int | None = 2  # Target reward used for early stopping
    learning_starts: int = 10  # Begin learning after this many steps
    gamma: float = 0.99  # Discount factor
    lr: float = 0.001  # Learning rate
    hidden_dim: int = 64  # Actor and critic network hidden dim
    buffer_capacity: int = 50_000  # Maximum replay buffer capacity
    batch_size: int = 32 * 2  # Batch size used by learner
    num_steps: int = 1  # Number of steps to unroll Bellman equation by
    tau: float = 0.005  # Soft target network update interpolation coefficient
    grad_norm_clip: float = 1000.0  # Global gradient clipping value

    noise_sigma: float = 0.50  # OU noise standard deviation
    sigma_decrease_coef: float = 0.01
    min_noise_sigma: float = 0.01

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

        if gl.load_neural_network:
            checkpoint_data = torch.load("models/DDPG_TF2-missile-learner_30000.pth")
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

                if self.iteration % 500 == 0:
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
            if step > self.config.learning_starts:
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

        # Training ended
        if self.config.verbose:
            print("\nTraining complete.")
        shared_collector.save_data()

        return logger.logs


def main():
    user_listener = UserListener()
    user_listener.start()

    # start program
    gl.start_program.wait()

    enviroment = None
    if gl.enviroment_type == "dummy":
        enviroment = DummyEnviroment()
    elif gl.enviroment_type == "normal":
        enviroment = Enviroment()
    else:
        enviroment = Enviroment()

    ddbg = DDPG(enviroment)
    ddbg.train()

    user_listener.stop()


if __name__ == "__main__":
    main()
