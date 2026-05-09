import numpy as np
import torch
import time
import os
import random
from datetime import datetime
from queue import Queue
import threading

import logger as lg
import globals as gl
import squirrel_api as sq
import TfBot as tf
from data_collector import shared_collector


class CustomActionSpace:
    "Parameters that descibes our inputs and outputs"

    def __init__(self, high: np.ndarray, low: np.ndarray):
        self.low = low
        self.high = high
        self.shape = self.low.shape


class AdaptiveSigma:
    """Dynamically adjusts a sigma value based on a rolling average of rewards."""

    def __init__(
        self,
        initial_sigma: float,
        minimal_sigma: float,
        max_sigma: float,
        sigma_step: float,
        window_size: int = 1000,
        increase_threshold: float = 0.01,
        decrease_threshold: float = 0.8,
    ):
        self.sigma = initial_sigma
        self.minimal_sigma = minimal_sigma
        self.max_sigma = max_sigma
        self.sigma_step = sigma_step
        self.window_size = window_size
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.recent_rewards = []

    def update(self, current_reward: float):
        self.recent_rewards.append(current_reward)
        if len(self.recent_rewards) >= self.window_size:
            rolling_mean = sum(self.recent_rewards) / self.window_size

            if rolling_mean > self.decrease_threshold:
                self._adjust_sigma(-self.sigma_step, "decreased")
            elif rolling_mean < self.increase_threshold:
                self._adjust_sigma(self.sigma_step, "increased")
            else:
                self.recent_rewards.pop(0)

    def _adjust_sigma(self, step: float, action_name: str):
        new_sigma = max(self.minimal_sigma, min(self.max_sigma, self.sigma + step))
        if new_sigma != self.sigma:
            self.sigma = new_sigma
            lg.logger.info(f"Sigma {action_name} to {self.sigma:.3f}")

            shared_collector.append("Sigma_change", f"{self.sigma:.3f}")
        self.recent_rewards.clear()


class Environment:
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

        self.adaptive_sigma = AdaptiveSigma(
            initial_sigma=1.0,
            minimal_sigma=0.1,
            max_sigma=1.0,
            sigma_step=0.05,
            decrease_threshold=0.7,
        )

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

        lg.logger.info(f"Iteration:{iteration}")

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
                rewards[i] = 1.0
            else:
                # sigma tunable: 0.3 ≈ 300 units, adjust to target hitbox size
                sigma = self.adaptive_sigma.sigma

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

        self.adaptive_sigma.update(rewards.mean().item())

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
        Environment.create_dir(self.logger_dir_name)

        Environment.save_data_to_file(
            str(self.accuracy_logger), "statistics_and_data/" + self.logger_dir_name + "/accuracy"
        )
        Environment.save_data_to_file(
            str(self.avg_reward_logger),
            "statistics_and_data/" + self.logger_dir_name + "/avg_reward",
        )
        Environment.save_data_to_file(
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
