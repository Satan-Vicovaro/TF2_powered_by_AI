import datetime
import os
from queue import Queue
import random
import numpy as np
import torch
import TfBot as tf

import logger as lg
from squirrel_api import tf2_listener_and_sender


class CustomActionSpace:
    "Parameters that descibes our inputs and outputs"

    def __init__(self, high: np.ndarray, low: np.ndarray):
        self.low = low
        self.high = high
        self.shape = self.low.shape


class Enviroment:
    s_bot_count = 20

    def __init__(self):
        # self.bots: dict[np.int64, tf.TfBot] = dict()  # all of our bots
        # self.t_bots: dict[np.int64, tf.TfBot] = dict()  # sub category: target bots
        # self.s_bots: dict[np.int64, tf.TfBot] = dict()  # sub category: shooter bots
        self.s_bots: torch.Tensor = torch.zeros([self.s_bot_count, 3])
        self.t_bot: torch.Tensor = torch.zeros([1, 3])
        self.player_input_messages: Queue = Queue()
        self.restart_count = 0
        self.iteration = 0
        self.accuracy_logger = []
        self.avg_reward_logger = []
        self.sum_reward_logger = []
        self.logger_dir_name = datetime.datetime.now().strftime("%H:%M")

    def __del__(self):
        os._exit(0)

    def get_observation_and_action_spaces(self):
        action_space = CustomActionSpace(
            high=np.array([360, 89]), low=np.array([0, -89])
        )  # -89 is maximum up position
        observation_space = CustomActionSpace(
            high=np.array([100, 100, 100, 100, 100, 100]),
            low=np.array([-100, -100, -100, -100, -100, -100]),
        )
        return action_space, observation_space

    def reset(self):
        """
        Resets our enviroment and gets initial position,
        in our case we just send next random position
        """
        self.s_bots = torch.randn_like(self.s_bots)

        # target bot randomization
        self.t_bot = torch.rand_like(self.t_bot)
        multipliers = torch.tensor([2, 2, 2])
        shifts = torch.tensor([-1.0, -1.0, -1.0])
        self.t_bot = (self.t_bot * multipliers) + shifts

        target_matrix = self.t_bot.repeat([self.s_bot_count, 1])
        observations = torch.cat((self.s_bots, target_matrix), dim=1)
        return observations

    def random_action(self):
        angles = torch.rand([self.s_bots.size(dim=0), 2])
        # print(angles)
        multipliers = torch.tensor([2, 2])
        shifts = torch.tensor([-1, -1])
        angles = (angles * multipliers) + shifts
        # print(angles)
        return angles

    def step(self, angles, observations, iteration):
        """
        the evaluation of the function,
        returns:
        next_obsevation (next bots positions, it does not change in our case),
        reward,
        terminated (if we succeed in achieving goal),
        truncated (timeout limit in training session)
        """

        multipliers = torch.tensor([180.0, 90.0])
        shifts = torch.tensor([180.0, 0.0])
        real_angles = (angles * multipliers) + shifts

        rewards = self.evaluate(real_angles, observations)
        # print(observations)
        # print(rewards)
        # our observations are 1 step only
        next_observation = self.reset()

        return (
            next_observation,
            rewards,
            torch.zeros_like(rewards, dtype=torch.bool),
            torch.zeros_like(rewards, dtype=torch.bool),
        )

    # def reset_damage_dealt(self):
    #     for bot in self.bots.values():
    #         bot.damage_dealt = np.float32(0.0)

    def evaluate(self, angles: torch.Tensor, observations: torch.Tensor):

        # proper evaluation function which uses ⭐️real math⭐️ to calculate proper rewards
        rewards = torch.zeros((angles.shape[0]))

        s_pos, t_pos = observations.split(3, dim=1)

        yaw_rad = torch.deg2rad(angles[:, 0])
        pitch_rad = torch.deg2rad(angles[:, 1])

        # Convert Spherical Angles to a 3D Direction Vector (d)
        # Note: Udjust it to the TF2 coordinates
        d_x = torch.cos(yaw_rad) * torch.cos(pitch_rad)
        d_y = torch.sin(yaw_rad) * torch.cos(pitch_rad)
        d_z = torch.sin(pitch_rad)

        v_missile_direction = torch.stack([d_x, d_y, d_z], dim=1)

        v_shooter_target = t_pos - s_pos  # vector: shooter ---> target

        dot_product_md_st = torch.sum(v_missile_direction * v_shooter_target, dim=1, keepdim=True)

        # adjusting for shooting backward
        dot_product_md_st = torch.clamp(dot_product_md_st, min=0.0)

        closest_point = s_pos + (dot_product_md_st * v_missile_direction)

        distances = torch.norm(t_pos - closest_point, dim=1)

        sigma = 0.4  # tunable
        # bell curve
        rewards = torch.exp(-(distances**2) / sigma**2)

        self.show_and_update_logs(rewards)
        return rewards

    def show_and_update_logs(self, rewards: torch.Tensor):

        lg.logger.debug(rewards)

        lg.logger.info("Average reward: {0:.2f}".format(rewards.mean()))
        self.avg_reward_logger.append("{0:.2f}".format(rewards.mean()))

        lg.logger.info("Sum of rewards: {0:.2f}".format(rewards.sum()))
        self.sum_reward_logger.append("{0:.2f}".format(rewards.sum()))

        # hit_counter = 0
        # for s_bot in self.s_bots.values():
        #     if s_bot.damage_dealt > 0:
        #         hit_counter += 1
        #
        # lg.logger.info("Accuracy: {0:.2f}".format(hit_counter / len(self.s_bots)))
        # self.accuracy_logger.append("{0:.2f}".format(hit_counter / len(self.s_bots)))
