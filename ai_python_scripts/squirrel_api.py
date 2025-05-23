from queue import Queue
import numpy as np
import logger as lg
import globals as gl
import os
import time
import TfBot as tf

#replace those lines with proper paths:
SQUIRREL_IN_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_in" 
SQUIRREL_OUT_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_out"

def handle_squirrel_input(player_input_messages:Queue):
    if player_input_messages.empty() :     
        return
    
    lg.logger.debug("tf2_listener: sending message to squirrel")
    #looking at Python OUT file
    with open(SQUIRREL_IN_PATH, "w+") as out_file: #squirrel_in == python_out
        #message format: message_type (string) | data (optional)

        while(not player_input_messages.empty()):
            out_file.write(player_input_messages.get() + "\n")


def handle_squirrel_output(bots:dict[np.int64,tf.TfBot]):
    
    #if file is empty
    # file is empty or have one null byte (Squirrel bug) 
    if not (os.stat(SQUIRREL_OUT_PATH).st_size > 1): 
        return

    #looking at IN file
    #squirrel_out == python_in 
    with open(SQUIRREL_OUT_PATH, 'r+') as in_file: 
        # formats: 
        # p (postion) t/s (target/shooter) bot_id pos_x pos_y pos_z
        # d (damage) bot_id damage 
          
        lg.logger.debug("tf2_listener: received output from squirrel")
        
        lines = in_file.readlines()

        #removing last character of last line (this buggy nullbyte)
        if lines and lines[-1]:  
            lines[-1] = lines[-1][:-1]
        #if damage was not dealed
        if lines[0] == "none" :
            gl.received_damage_data.set()
            in_file.truncate(0) # clearing contents
            return

        position_data = False
        damage_data = False
        
        for line in lines:
            parts = line.split(sep=" ")

            
            #mparts[0] = message type
            match parts[0]: 
                case"p":
                    position_data = True
                    #postion
                    bot_type = parts[1]
                    bot_id = np.int64(parts[2])
                    pos_x = np.float64(parts[3])
                    pos_y = np.float64(parts[4])
                    pos_z = np.float64(parts[5])

                    if bot_id in bots:
                        #update existing one
                        bots[bot_id].pos_x = pos_x                        
                        bots[bot_id].pos_y = pos_y                        
                        bots[bot_id].pos_z = pos_z                        
                    else:
                        #create if it does not exit
                        bots[bot_id] = tf.TfBot(pos_x = pos_x, pos_y = pos_y,pos_z = pos_z, bot_type = bot_type)

                case"d": 
                    damage_data = True
                    bot_id = np.int64(parts[1])
                    damage = np.float64(parts[2])
                    if not bot_id in bots:
                        lg.logger.error("Error: Bot with id: " + str(bot_id) + " does not exist")                    
                    bots[bot_id].damage_dealt = damage

                case"_":
                    lg.logger.error("Error: why start of message is: " + parts[0])


        if damage_data and position_data:
            lg.logger.error("Error? : I got damage_data and postion_data in 1 message")
        elif damage_data:
            gl.received_damage_data.set()
        elif position_data:
            gl.received_positions_data.set()

        in_file.truncate(0) # clearing contents
    

def tf2_listener_and_sender(player_input_messages:Queue, bots:dict[np.int64,tf.TfBot]):
    lg.logger.info("tf2_listener: Listening on tf2 files")

    #clearing in and out files
    with open(SQUIRREL_OUT_PATH, 'w') as in_file: 
        in_file.truncate(0)

    with open(SQUIRREL_IN_PATH, 'w') as out_file: 
        out_file.truncate(0)
    

    while not gl.end_program.is_set():
        #input
        handle_squirrel_output(bots)

        #if we dont have antyhing to send
        if not gl.send_message.is_set():
            time.sleep(gl.POLLING_INTERVAL)
            continue

        #output
        handle_squirrel_input(player_input_messages)
        
        #waiting for next check
        time.sleep(gl.POLLING_INTERVAL)
    
    
    #sending exit command to python_listener in tf2
    with open(SQUIRREL_IN_PATH, "w") as out_file:
        out_file.write("exit |")
