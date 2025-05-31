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

def display_troch_data(angles:torch.Tensor,rewards:torch.Tensor):
    lg.logger.debug("angles: " + str(angles))
    lg.logger.debug("rewards: " + str(rewards))

def display_hit_data(bots:dict[np.int64,tf.TfBot]): 
    lg.logger.info("Bot damage data: ")
    for bot_id, bot in zip(bots.keys(), bots.values()):
        if bot.damage_dealt != 0:
            lg.logger.info("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
 
        #reseting damage_dealt!!
        bot.damage_dealt = 0


def collect_tracker_data(s_bots:dict[np.int64, tf.TfBot], rewards: torch.Tensor, overall_evaluation_tracker, accuracy_tracker):
    hits = 0
    for s_bot in s_bots.values():
        if s_bot.damage_dealt != 0:
            hits += 1

    accuracy = float(hits/len(s_bots)) 
    reward_sum = sum(rewards) 

    lg.logger.info("Sum of rewards: " + str(reward_sum))
    lg.logger.info("Accuracy: " + str(accuracy))
    
    overall_evaluation_tracker.append(int(reward_sum))
    accuracy_tracker.append(accuracy)


def send_wait_for_bullet_data(restart_count, player_input_messages:Queue):
    # requesting bullets distances from target_bot
    player_input_messages.put("send_distances|")
    gl.send_message.set()

    lg.logger.debug("waiting for bullet data")
    # waiting for damage data
    if not gl.received_bullet_data.wait(gl.MAX_DURATION):
        lg.logger.warning("Received bullet data timeout reached, restarting the loop...")
        restart_count += 1
        lg.logger.warning("Restart count: " + str(restart_count))
        return True, restart_count
    gl.received_bullet_data.clear() # don't forget to clear the flag
    return False, restart_count

def send_and_wait_for_positions(restart_count,player_input_messages:Queue): 

    #request data postion data
    player_input_messages.put("get_position |") #alway end message_type with "|"
    gl.send_message.set()

    lg.logger.debug("Waiting for positions")
    #waiting for positions
    if not gl.received_positions_data.wait(timeout = gl.MAX_DURATION):
        lg.logger.warning("Received position timeout reached, restarting the loop...")
        restart_count += 1
        lg.logger.warning("Restart count: " + str(restart_count))
        return True, restart_count
    gl.received_positions_data.clear() #removing flag
    return False,restart_count


def send_and_wait_for_damage_data( restart_count,player_input_messages:Queue):
    #requesting damage data
    player_input_messages.put( "send_damage|" )
    gl.send_message.set()

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
                # watiging for squirrel to init it self 
                time.sleep(0.25) 
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
    
    lg.logger.debug("Sending angles")
    
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

    result = torch.zeros((angles.shape[0]))
    for i,s_bot in enumerate(s_bots.values()):
        #dont shoot into celing
        if angles[i][0] > 70.0:
            result[i] -= torch.tensor(abs((angles[i][0]  - 40.0)) * 35.0 )
        
        #dont shoot into ground
        if angles[i][0] < -70:
            result[i] -= torch.tensor(abs((angles[i][0] + 40.0)) * 35.0)
        
        result[i] +=  torch.tensor(-(s_bot.m_distance * 0.01)**2 + s_bot.damage_dealt)

    return result


def evaluate_decision(s_bot: tf.TfBot, t_bot:tf.TfBot ):
    p1 = np.array([s_bot.pos_x, s_bot.pos_y, s_bot.pos_z])
    p2 = np.array([t_bot.pos_x, t_bot.pos_y, t_bot.pos_z])
    return torch.tensor(-np.linalg.norm(p1 - p2))

def training_loop(bots:dict[np.int64, tf.TfBot], player_input_messages, actor:ai.Actor, actor_optimizer:ai.optim, replay_buffer:ai.ReplayBuffer, overall_evaluation_tracker, accuracy_tracker):
    global restart_count

    should_restart, restart_count = send_and_wait_for_positions(restart_count, player_input_messages)
    if should_restart:
        return        
        
    normalize_data(bots)

    t_bots,s_bots = dispatch_bots_into_shooters_and_targets(bots)
        
    # training_data = dict: int -> torch.Tensor
    training_data = crate_training_data(t_bots,s_bots)  

    # we dont extract exact values form out neural network
    # but we get:
    # - mean (most likely action)
    # - std (standard deviation of actions)
    # we need them to create proper griadient for evaluation
    means, stds = actor(training_data)

    noise_scale = 0.4  # tune this
    noisy_means = means + torch.randn_like(means) * noise_scale

    dist = torch.distributions.Normal(noisy_means, stds)

    # now we get angles:
    # angles = Tensor[s_bot_num, 2]    
        
    # angles = mean  #<- this would be the most likley action 
    angles = dist.sample() # this is some action based from propability

    angles[:, 0] = torch.clamp(angles[:, 0], -89, 89)
    angles[:, 1] = torch.clamp(angles[:, 1], -179, 179)    


    #squashed = torch.tanh(raw_angle_sample)  # in [-1, 1]
       
    #scales = torch.tensor([179.0, 89], device=squashed.device)
    #angles = squashed * scales  # now in [-179, 179] and [-89, 89] 

    send_tensor_angles(player_input_messages, angles, s_bots)

    # wait for damage response
    time.sleep(2.0)

    should_restart, restart_count = send_and_wait_for_damage_data(restart_count, player_input_messages)
    if should_restart:
        return
        
    should_restart, restart_count = send_wait_for_bullet_data(restart_count, player_input_messages)
    if should_restart:
        return
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
    rewards_normal = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    log_probs = dist.log_prob(angles).sum(dim=1)

    loss = -(log_probs * rewards_normal).mean()

    actor_optimizer.zero_grad() # reseting gradient
    loss.backward()
    actor_optimizer.step()


    replay_buffer.update_queue(s_bots,t_bot,angles,rewards) 
    replay_buffer.update_file_csv()
    replay_buffer.update_file_pickle()
    collect_tracker_data(s_bots, rewards, overall_evaluation_tracker, accuracy_tracker )
    display_hit_data(bots)
    display_troch_data(angles,rewards)
    reset_damage_dealt(bots)
        

def math_magic(s_x, s_y, s_z, t_x, t_y, t_z):
    # thats almost what we want from our neural network...
    a = np.array([s_x, s_y, s_z])
    b = np.array([t_x, t_y ,t_z])
    
    dir_vec = a - b
 
    x, y, z = dir_vec
    
    # Yaw angle (around Z-axis), angle in XY plane from X-axis
    yaw = np.arctan2(y, x)
    
    # Pitch angle (around Y-axis), angle from horizontal XY plane to vector
    hyp = np.sqrt(x**2 + y**2)
    pitch = np.arctan2(z, hyp)
    
    # to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    
    return pitch_deg, yaw_deg


def random_loop(player_input_messages, bots, replay_buffer:ai.ReplayBuffer):
    global restart_count, iteration

    should_restart, restart_count = send_and_wait_for_positions(restart_count, player_input_messages)
    if should_restart:
        return       

     
    t_bots,s_bots = dispatch_bots_into_shooters_and_targets(bots)

    #getting target bot from dictionary
    t_bot:tf.TfBot = next(iter(t_bots.values()))

    angles = torch.zeros((len(s_bots)), 2)
    # HERE WE PUT OUR GREAT AI MODEL
    for i, s_bot in enumerate(s_bots.values()):
        pitch, yaw = math_magic(s_bot.pos_x,s_bot.pos_y,s_bot.pos_z, t_bot.pos_x, t_bot.pos_y, t_bot.pos_z)
        
        #yes, pitch and yaw are asigned wrongly in the whole code
        
        s_bot.pitch = float((yaw +180)+ random.uniform(-5 , 5))      
        s_bot.yaw =  pitch  + random.uniform(-2.5, 2.5) 
        angles[i] = torch.tensor([s_bot.pitch, s_bot.yaw])
    lg.logger.debug("Sending angles")
    send_angles(bots, player_input_messages) 
    
    # wait for damage response
    time.sleep(1.2)

    should_restart, restart_count = send_and_wait_for_damage_data(restart_count, player_input_messages)
    if should_restart:
        return

     

    replay_buffer.update_queue(s_bots,t_bot, angles)
    replay_buffer.update_file_csv()
    display_hit_data(bots) 
    reset_damage_dealt(bots)

    if iteration % 2 == 0:
        player_input_messages.put("change_target_pos|") 
        gl.send_message.set()
        time.sleep(0.4)
    else:
       player_input_messages.put("change_shooter_pos|")
       gl.send_message.set()
       time.sleep(0.4)

iteration = 0
restart_count = 0
def main():
    global iteration, restart_count
    bots: dict[np.int64,tf.TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = sq.tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()
    
    # ai setup
    actor = ai.actor
    actor_optimizer = ai.actor_optimizer

    replay_buffer = ai.ReplayBuffer()


    overall_evaluation_tracker = []
    accuracy_tracker = []
    # loop
    #replay_buffer.load_form_file_pickle()
    #replay_buffer.update_file_csv()
    while not gl.end_program.is_set():
        
        #start program
        gl.start_program.wait()    

        lg.logger.info("iteration: " + str(iteration))
        
        #training_loop(bots,player_input_messages, actor, actor_optimizer, replay_buffer, overall_evaluation_tracker, accuracy_tracker)

        random_loop(player_input_messages, bots, replay_buffer)

        iteration += 1
         

 
    with open("statistics_and_data/Accuracy.txt", "w") as file:
        file.write(str(accuracy_tracker))
   
    with open("statistics_and_data/Rewards_sum.txt", "w") as file:
        file.write(str(overall_evaluation_tracker))


    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()