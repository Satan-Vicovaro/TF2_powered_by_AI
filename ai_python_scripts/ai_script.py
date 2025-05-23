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


def send_angles(bots: dict[np.int64,tf.TfBot], player_input_messages: Queue):

    message = "angles |"

    # message format:
    # bot_id pitch (y) yaw (x)
    for bot_id,bot in zip (bots.keys(), bots.values()):
        message += (" {0} {1} {2}\n".format(bot_id, bot.pitch, bot.yaw))

    player_input_messages.put(message)
    gl.send_message.set()



def main():
    
    bots: dict[np.int64,tf.TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = sq.tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()
    
    iteration = 0
    restart_count = 0
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


        # HERE WE PUT OUR GREAT AI MODEL
        for bot in bots.values():
            bot.pitch = random.uniform(-179,179)
            bot.yaw =  random.uniform(-89,89) # yaw < 0  = up

        lg.logger.debug("Sending angles")
        send_angles(bots, player_input_messages) 
        
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
        
        lg.logger.info("Bot damage data: ")
        for bot_id, bot in zip(bots.keys(), bots.values()):
            if bot.damage_dealt != 0:
                lg.logger.info("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
 
            #reseting damage_dealt!!
            bot.damage_dealt = 0

        iteration += 1
        #loop ends ig
        
    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()