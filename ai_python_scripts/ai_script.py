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

import torch
import torch.nn as nn
import TfBot
import cma

class AnglePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 32), # input
            nn.ReLU(), #input into ReLU transformation

            nn.Linear(32, 16), #middle one
            nn.ReLU(), #into Relu

            nn.Linear(16, 2),  #into pitch and yaw
            nn.Tanh() # transforming our output into values between -1 and 1
        )

    def forward(self, x):
        return self.model(x)


#this function should be gradient-like
def evaluate_bot(s_bot:tf.TfBot, angles:torch.Tensor):
    
    #advicing bots to shoot somewere in the air
    target = - 0.4
    diff = abs(float(angles[1]) - target)
    value = float(max(-1, 10 - diff * 25))

    if -0.6 < angles[1] < 0.8:
        value += float(random.uniform(2,10))


    if s_bot.damage_dealt > 0:
        return -float(value + s_bot.damage_dealt + 10.0)
    else:
        return -float(value)

def evaluate_the_batch(s_bots:dict[np.int64,tf.TfBot]):
    #this evaluation is lame, bcs AI does not know which batch smple was correct
    sum = []

    for bot in s_bots.values():
        sum.append(bot.damage_dealt)
        bot.damage_dealt = 0
    
    return sum

model = AnglePredictor()

#flattens our model to 2 dim bcs CMA-ES need that
def get_flat_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])


#form flat into proper arr
def set_flat_params(model, flat_params):
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer+numel].view_as(param))
        pointer += numel



def fitness(flat_weights,training_data):

    set_flat_params(model, torch.tensor(flat_weights, dtype=torch.float32))
    total_score = 1.0
    for input_vector in training_data:
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        pred_angles = model(input_tensor)
        total_score += model.evaluate(input_tensor, pred_angles)
    return -total_score


def normalize_data(bots: dict[np.int64,tf.TfBot]):
    for bot in bots.values():
        bot.normalize()

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


def crate_training_data(t_bots: dict[np.int64,tf.TfBot], s_bots: dict[np.int64,tf.TfBot]):

    if len(t_bots) > 1:
        lg.logger.warning("We have more than one target bot")

    target_bot:tf.TfBot = next(iter(t_bots.values()))
    
    result = torch.tensor([(bot.pos_x, bot.pos_y, bot.pos_z, target_bot.pos_x, target_bot.pos_y,target_bot.pos_z) for bot in s_bots.values() ],dtype=torch.float32 )
    print(result.shape)
    return result 
    


def user_input_listener(player_input_messages:Queue):
    lg.logger.info("listening for player input") 

    while not gl.end_program.is_set():
        try:
            user_input = input("[You] > ").strip()
            if user_input.lower() == "exit":
                gl.end_program.set()
            elif user_input.lower() == "start":
                player_input_messages.put("start |")
                gl.start_program.set()
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



def send_tensor_angles(player_input_messages: Queue, angles:dict[np.int64, torch.Tensor]):

    message = "angles |" 
    
    # message format:
    # bot_id pitch (y) yaw (x)
    for bot_id, values in angles.items():
        message += (" {0} {1} {2}\n".format(bot_id,179 * values[0], 89 * values[1]))

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


BATCH_SIZE = 10

def main():
    
    bots: dict[np.int64,tf.TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = sq.tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()
    
    iteration = 0
    restart_count = 0
    
    # ai setup 
    initial_weights = get_flat_params(model).detach().numpy()
    es = cma.CMAEvolutionStrategy(initial_weights, 0.1)
    cma._warnings.filterwarnings('ignore', message='The number of solutions passed to `tell` should.*') # disabling waring
    

    while not gl.end_program.is_set():


        #start program
        gl.start_program.wait()    

        # watiging for squirrel to init it self 
        time.sleep(0.25) 
                
        lg.logger.info("iteration: " + str(iteration))
        #request data postion data
        player_input_messages.put("get_position |") #alway end message_type with "|"
        gl.send_message.set()


        lg.logger.debug("Waiting for positions")
        #waiting for positions
        if not gl.received_positions_data.wait(timeout = gl.MAX_DURATION):
            lg.logger.warning("Received position timeout reached, restarting the loop...")
            restart_count += 1
            lg.logger.warning("Restart count: " + str(restart_count))
            continue

        gl.received_positions_data.clear() #removing flag
        
        # now we have updated dictionary
        normalize_data(bots)

        # we split them into 2 categories
        target_bots, shooter_bots = dispatch_bots_into_shooters_and_targets(bots)
        

        if len(target_bots) > 1:
            lg.logger.warning("We have more than one target bot")

        t_bot:tf.TfBot = next(iter(target_bots.values()))

        # Ask CMA-ES for candidate solutions
        solutions = es.ask(number = len(shooter_bots))

        angles:dict[np.int64, torch.Tensor] = {}

        for i, (bot_id,s_bot) in enumerate(shooter_bots.items()):
            
            # Apply weights to model
            set_flat_params(model, torch.tensor(solutions[i], dtype=torch.float32))
            
            # load one bot to model
            angles[bot_id] = model(torch.tensor([s_bot.pos_x,s_bot.pos_y,s_bot.pos_z,
                                t_bot.pos_x,t_bot.pos_y,t_bot.pos_z], dtype = torch.float32))
            lg.logger.debug("Bot_input: x:{} y:{} z:{} x:{} y:{} z:{}".format(s_bot.pos_x,s_bot.pos_y,s_bot.pos_z,
                                                                              t_bot.pos_x,t_bot.pos_y,t_bot.pos_z))

        # now all bots have angles chosen by AI

        #lg.logger.debug(angles)
        lg.logger.debug("Sending angles")
        send_tensor_angles(player_input_messages, angles)

        # wait for damage response
        time.sleep(3.0)

        #requesting damage data
        player_input_messages.put( "send_damage|" )
        gl.send_message.set()


        lg.logger.debug("waiting for damage data")

        # waiting for damage data
        if not gl.received_damage_data.wait(gl.MAX_DURATION):
            lg.logger.warning("Received damage data timeout reached, restarting the loop...")
            restart_count += 1
            lg.logger.warning("Restart count: " + str(restart_count))
            continue
        gl.received_damage_data.clear() # don't forget to clear the flag


        scores = []

        for bot_id,s_bot in shooter_bots.items():
            score = evaluate_bot(s_bot, angles[bot_id])
            scores.append(score)
            s_bot.damage_dealt = 0
        

        iteration += 1
        es.tell(solutions,scores)
        lg.logger.info(scores)  

        if iteration % 50 == 0:
            player_input_messages.put("change_shooter_pos |")
        
        #if iteration % 30 == 0:
        #    player_input_messages.put("change_target_pos |")
        # for flat_weights in solutions:

        #     # Apply weights to model
        #     set_flat_params(model, torch.tensor(flat_weights, dtype=torch.float32))
            
        #     # Prepare batch
        #     training_batch = crate_training_data(target_bots,shooter_bots) # translating dict bots into torch.tensor array 

        #     angle_batch = model(training_batch) # <- that line does all the heavy work ig


        #     lg.logger.debug("Sending angles")
        #     send_tensor_angles(shooter_bots, player_input_messages, angle_batch)

        #     # wait for damage response
        #     time.sleep(3.0)

            
        #     #requesting damage data
        #     player_input_messages.put( "send_damage|" )
        #     gl.send_message.set()

        #     lg.logger.debug("waiting for damage data")
        #     # waiting for damage data
        #     if not gl.received_damage_data.wait(gl.MAX_DURATION):
        #         lg.logger.warning("Received damage data timeout reached, restarting the loop...")
        #         restart_count += 1
        #         lg.logger.warning("Restart count: " + str(restart_count))
        #         continue
        #     gl.received_damage_data.clear() # don't forget to clear the flag


        #     reward = evaluate_the_batch(shooter_bots)
        #     scores.append(reward)

        #     iteration += 1

        # es.tell(solutions,scores)

        # # HERE WE PUT OUR GREAT AI MODEL
        # for bot in bots.values():
        #     bot.pitch = random.uniform(-179,179)
        #     bot.yaw =  random.uniform(-89,89) # yaw < 0  = up

        # lg.logger.debug("Sending angles")
        # send_angles(bots, player_input_messages) 
        
        # # wait for damage response
        # time.sleep(3.0)

        # #requesting damage data
        # player_input_messages.put( "send_damage|" )
        # gl.send_message.set()

        # lg.logger.debug("waiting for damage data")
        # # waiting for damage data
        # if not gl.received_damage_data.wait(gl.MAX_DURATION):
        #     lg.logger.warning("Received damage data timeout reached, restarting the loop...")
        #     restart_count += 1
        #     lg.logger.warning("Restart count: " + str(restart_count))
        #     continue
        # gl.received_damage_data.clear() # don't forget to clear the flag
        
        # lg.logger.info("Bot damage data: ")
        # for bot_id, bot in zip(bots.keys(), bots.values()):
        #     if bot.damage_dealt != 0:
        #         lg.logger.info("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
 
        #     #reseting damage_dealt!!
        #     bot.damage_dealt = 0

        # iteration += 1
        # #loop ends ig
        
    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()