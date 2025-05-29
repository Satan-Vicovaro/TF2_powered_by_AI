import random
import threading
import time
import os
from queue import Queue
import numpy as np
from enum import Enum, auto

import logger as lg
import squirrel_api as sq
import globals as gl
import TfBot as tf

import ai_module as ai

import torch
import torch.nn as nn

def reset_damage_dealt(bots:dict[np.int64,tf.TfBot]):
    for bot in bots.values():
        bot.damage_dealt = 0

def display_hit_data(bots:dict[np.int64,tf.TfBot]): 
    lg.logger.info("Bot damage data: ")
    for bot_id, bot in zip(bots.keys(), bots.values()):
        if bot.damage_dealt != 0:
            lg.logger.info("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
 
        #reseting damage_dealt!!
        bot.damage_dealt = 0


def wait_for_positions(restart_count): 
    lg.logger.debug("Waiting for positions")
    #waiting for positions
    if not gl.received_positions_data.wait(timeout = gl.MAX_DURATION):
        lg.logger.warning("Received position timeout reached, restarting the loop...")
        restart_count += 1
        lg.logger.warning("Restart count: " + str(restart_count))
        return True, restart_count
    gl.received_positions_data.clear() #removing flag
    return False,restart_count


def wait_for_damage_data( restart_count):
    lg.logger.debug("waiting for damage data")
    if not gl.received_damage_data.wait(gl.MAX_DURATION):
        lg.logger.warning("Received damage data timeout reached, restarting the loop...")
        restart_count += 1
        lg.logger.warning("Restart count: " + str(restart_count))
        return True, restart_count  
    gl.received_damage_data.clear()
    return False, restart_count  


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
                #waiting for squirrel to init before messaging main python loop
                time.sleep(0.5)
                gl.start_program.set()
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



def send_tensor_angles(player_input_messages: Queue, angles:torch.Tensor, s_bots: dict[np.int64, tf.TfBot]):

    message = "angles |" 
    
    # message format:
    # bot_id pitch (y) yaw (x)
    for i,bot_id in enumerate(s_bots.keys()):
        message += (" {0} {1} {2}\n".format(bot_id, angles[i][0], angles[i][1]))

    player_input_messages.put(message)
    gl.send_message.set()



def send_angles(bots: dict[np.int64,tf.TfBot], player_input_messages: Queue):

    message = "angles |"

    # message format:
    # bot_id pitch (y) yaw (x)
    for bot_id,bot in zip (bots.keys(), bots.values()):
        message += (" {0} {1} {2}\n".format(bot_id, bot.pitch, bot.yaw))

    player_input_messages.put(message)
    gl.send_message.set()


def normalize_data(bots: dict[np.int64,tf.TfBot]):
    for bot in bots.values():
        bot.normalize()
    

def crate_training_data(t_bots: dict[np.int64,tf.TfBot], s_bots: dict[np.int64,tf.TfBot]): 

    if len(t_bots) > 1:
        lg.logger.warning("We have more than one target bot")

    t_bot:tf.TfBot = next(iter(t_bots.values()))

    
    return torch.tensor([(bot.pos_x, bot.pos_y, bot.pos_z, t_bot.pos_x, t_bot.pos_y,t_bot.pos_z) for bot in s_bots.values()], dtype=torch.float32)



def dispatch_bots_into_shooters_and_targets(bots: dict[np.int64,tf.TfBot]):

    target_bots: dict[np.int64,tf.TfBot] = {}
    shooter_bots: dict[np.int64,tf.TfBot] = {}

    # seperating shooters from target
    for key, bot in bots.items():
        if bot.bot_type == "s":
            shooter_bots[key] = bot
            continue
        if bot.bot_type == "t":
            target_bots[key] = bot        
            continue

        lg.logger.warning("There is a bot without BotType?\n",  bot)
    return target_bots,shooter_bots

def evaluate_all_decisions(angles: torch.Tensor, s_bots:dict[np.int64,tf.TfBot], t_bot: tf.TfBot):
    # Current evaluation:
    # eval = -distance(m_miss , t_bot.pos) + damage_dealt    
    
    #p1 = np.array([t_bot.pos_x, t_bot.pos_y, t_bot.pos_z])

    target = -45.0
    

    result = torch.zeros((angles.shape[0]))
    for i,s_bot in enumerate(s_bots.values()):
#       p2 = np.array([s_bot.m_miss_x, s_bot.m_miss_y, s_bot.m_miss_z])
#       result[i] = - torch.tensor(np.linalg.norm( p1 - p2 ) + s_bot.damage_dealt)

        diff = abs(float(angles[i][1]) - target)
        value = float(max(-20, 20 - diff * 25))
        result[i] = torch.tensor(value + s_bot.damage_dealt)
    return result


def evaluate_decision(s_bot: tf.TfBot, t_bot:tf.TfBot ):
    p1 = np.array([s_bot.pos_x, s_bot.pos_y, s_bot.pos_z])
    p2 = np.array([t_bot.pos_x, t_bot.pos_y, t_bot.pos_z])
    return torch.tensor(-np.linalg.norm(p1 - p2))


def main():
    
    bots: dict[np.int64,tf.TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = sq.tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()
    
    # ai setup
    actor = ai.actor
    actor_target = ai.actor_target
    
    critic = ai.critic
    critic_target = ai.critic_target

    actor_optimizer = ai.actor_optimizer
    critic_optimizer = ai.critic_optimizer

    replay_buffer = ai.replay_buffer

    state = ai.sample_random_state()
    episode_reward = 0

    # loop
    iteration = 0
    restart_count = 0
    while not gl.end_program.is_set():
        
        #start program
        gl.start_program.wait()    
 
        lg.logger.info("iteration: " + str(iteration))

        #request data postion data
        player_input_messages.put("get_position |") #alway end message_type with "|"
        gl.send_message.set()


        should_restart, restart_count = wait_for_positions(restart_count)
        if should_restart:
            continue        
        
        normalize_data(bots)
        t_bots,s_bots = dispatch_bots_into_shooters_and_targets(bots)
        
        # training_data = dict: int -> torch.Tensor
        training_data = crate_training_data(t_bots,s_bots)  

        # we dont extract exact values form out neural network
        # but we get:
        # - mean (most likely action)
        # - std (standard deviation)
        # we need them to create proper griadient for evaluation
        means, stds = actor(training_data)

        noise_scale = 0.2  # tune this
        noisy_means = means + torch.randn_like(means) * noise_scale

        dist = torch.distributions.Normal(noisy_means, stds)

        # now we get angles:
        # angles = Tensor[s_bot_num, 2]    
        
        # angles = mean  #<- this would be the most likley action 
        raw_angle_sample = dist.sample() # this is some action based from propability
        
        squashed = torch.tanh(raw_angle_sample)  # in [-1, 1]
       
        scales = torch.tensor([179.0, 89], device=squashed.device)
        angles = squashed * scales  # now in [-179, 179] and [-89, 89] 

        lg.logger.debug("Sending angles")
        send_tensor_angles(player_input_messages, angles, s_bots)

        # wait for damage response
        time.sleep(3.0)

        #requesting damage data
        player_input_messages.put( "send_damage|" )
        gl.send_message.set()

        should_restart, restart_count = wait_for_damage_data(restart_count)
        if should_restart:
            continue
        

        # evaluation
        # current option:
        # our_evaluation --teaches--> actor
        #
        # more fancy option:
        # our_evaluation --teaches--> critc --teaches--> actor

        #getting target bot from dictionary
        t_bot:tf.TfBot = next(iter(t_bots.values()))

        rewards = evaluate_all_decisions(angles,s_bots, t_bot)
        
        # Normalize rewards (optional but stabilizes training)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        log_probs = dist.log_prob(raw_angle_sample).sum(dim=1)

        loss = -(log_probs * rewards).mean()

        actor_optimizer.zero_grad() # resetig gradient
        loss.backward()
        actor_optimizer.step()

        display_hit_data(bots)
        reset_damage_dealt(bots)

        iteration += 1
        #loop ends ig
        
    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()