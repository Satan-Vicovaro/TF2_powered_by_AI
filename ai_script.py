import random
import threading
import time
import os
from queue import Queue
import numpy as np
from enum import Enum, auto
import logging
import colorlog
import colorama
import json


POLLING_INTERVAL = 0.01 # in seconds [s]
MAX_DURATION = 10 # seconds

SQUIRREL_IN_PATH = None
SQUIRREL_OUT_PATH = None

import json
from pathlib import Path

def load_existing_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"[config] Expected config file not found at: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)

def initialize_paths():
    global SQUIRREL_IN_PATH, SQUIRREL_OUT_PATH

    # Config file is one directory above the script
    config_path = Path.cwd().parent / "config.json"

    config = load_existing_config(config_path)

    if "scriptdata" not in config:
        raise KeyError("[config] 'scriptdata' key missing from config.json")

    scriptdata_path = Path(config["scriptdata"])
    SQUIRREL_IN_PATH = scriptdata_path / "squirrel_in"
    SQUIRREL_OUT_PATH = scriptdata_path / "squirrel_out"

    print("[paths] SQUIRREL_IN_PATH =", SQUIRREL_IN_PATH)
    print("[paths] SQUIRREL_OUT_PATH =", SQUIRREL_OUT_PATH)

#-----------------------------------logger + ⭐️ fancy colors ⭐️ ----------------------------------------------- 

colorama.init()  # Required on Windows to support ANSI colors

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s] %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red,bg_white',
    }
))

logger = logging.getLogger("ai_script")
logger.setLevel(logging.INFO)

# Create and attach a console handler
#formatter = logging.Formatter('[%(levelname)s] %(message)s')
#console.setFormatter(formatter)
#console = logging.StreamHandler()
logger.addHandler(handler)

# Function to toggle debug logging
def enable_debug():
    logger.setLevel(logging.DEBUG)
    logger.debug("Debugging enabled")

def disable_debug():
    logger.setLevel(logging.INFO)
    logger.info("Debugging disabled")


#------------------------------------ thread flags -----------------------------------
end_program = threading.Event()

start_program = threading.Event()

send_message = threading.Event()

received_positions_data = threading.Event()

received_damage_data = threading.Event()

received_bullet_data = threading.Event()

class BotType(Enum):
    NONE = 0
    SHOOTER = 't'
    TARGET = 's'


class TfBot: 
    def __init__(
            self,
            pos_x: float = 0.0,
            pos_y: float = 0.0,
            pos_z: float = 0.0,
            pitch: float = 0.0,
            yaw: float = 0.0,
            vel_x: float = 0.0,
            vel_y: float = 0.0,
            vel_z: float = 0.0,
            bot_type: BotType = BotType.NONE,
            damage_dealt: float = 0.0,
            bullet_distance: float = 0.0
        ):
            self.pos_x = np.float64(pos_x)
            self.pos_y = np.float64(pos_y)
            self.pos_z = np.float64(pos_z)

            self.pitch = np.float64(pitch)
            self.yaw = np.float64(yaw)

            self.vel_x = np.float64(vel_x)
            self.vel_y = np.float64(vel_y)
            self.vel_z = np.float64(vel_z)

            self.bot_type = bot_type
            self.damage_dealt = np.float64(damage_dealt)
            self.bullet_distance = np.float64(bullet_distance)





def handle_squirrel_input(player_input_messages:Queue):
    if player_input_messages.empty() :     
        return
    
    logger.debug("tf2_listener: sending message to squirrel")
    #looking at Python OUT file
    with open(SQUIRREL_IN_PATH, "w+") as out_file: #squirrel_in == python_out
        #message format: message_type (string) | data (optional)

        while(not player_input_messages.empty()):
            out_file.write(player_input_messages.get() + "\n")


def handle_squirrel_output(bots:dict[np.int64,TfBot]):
    
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
          
        logger.debug("tf2_listener: received output from squirrel")
        
        lines = in_file.readlines()

        #removing last character of last line (this buggy nullbyte)
        if lines and lines[-1]:  
            lines[-1] = lines[-1][:-1]
        #if damage was not dealed
        if lines[0] == "d none" :
            received_damage_data.set()
            in_file.truncate(0) # clearing contents
            return
        
        #if no distances provided
        if lines[0] == "b none" :
            received_bullet_data.set()
            in_file.truncate(0) # clearing contents
            return

        position_data = False
        damage_data = False
        bullet_data = False
        
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
                        bots[bot_id] = TfBot(pos_x = pos_x, pos_y = pos_y,pos_z = pos_z, bot_type = bot_type)
                
                case "b":
                    bullet_data = True
                    bot_id = np.int64(parts[1])
                    distance = np.float64(parts[2])
                    if not bot_id in bots:
                        logger.error("Error: Bot with id: " + str(bot_id) + " does not exist")  
                    bots[bot_id].bullet_distance = distance

                case"d": 
                    damage_data = True
                    bot_id = np.int64(parts[1])
                    damage = np.float64(parts[2])
                    if not bot_id in bots:
                        logger.error("Error: Bot with id: " + str(bot_id) + " does not exist")                    
                    bots[bot_id].damage_dealt = damage

                case"_":
                    logger.error("Error: why start of message is: " + parts[0])


        if damage_data and position_data:
            logger.error("Error? : I got damage_data and postion_data in 1 message")
        elif damage_data:
            received_damage_data.set()
        elif position_data:
            received_positions_data.set()
        elif bullet_data:
            received_bullet_data.set()

        in_file.truncate(0) # clearing contents
    

def tf2_listener_and_sender(player_input_messages:Queue, bots:dict[np.int64,TfBot]):
    logger.info("tf2_listener: Listening on tf2 files")

    #clearing in and out files
    with open(SQUIRREL_OUT_PATH, 'w') as in_file: 
        in_file.truncate(0)

    with open(SQUIRREL_IN_PATH, 'w') as out_file: 
        out_file.truncate(0)
    

    while not end_program.is_set():
        #input
        handle_squirrel_output(bots)

        #if we dont have antyhing to send
        if not send_message.is_set():
            time.sleep(POLLING_INTERVAL)
            continue

        #output
        handle_squirrel_input(player_input_messages)
        
        #waiting for next check
        time.sleep(POLLING_INTERVAL)
    
    
    #sending exit command to python_listener in tf2
    with open(SQUIRREL_IN_PATH, "w") as out_file:
        out_file.write("exit |")


def user_input_listener(player_input_messages:Queue):
    logger.info("listening for player input") 

    while not end_program.is_set():
        try:
            user_input = input("[You] > ").strip()
            if user_input.lower() == "exit":
                end_program.set()
            elif user_input.lower() == "start":
                player_input_messages.put("start |")
                start_program.set()
            elif user_input.lower() == "debug on": 
                enable_debug()
            elif user_input.lower() == "debug off": 
                disable_debug()
            else:
                player_input_messages.put(user_input + " |")  

            logger.debug("User_listener: Got input form player")
            send_message.set()          
                                
        except (KeyboardInterrupt,EOFError):
            logger.error("Keyboard error occured")
            end_program.set()


def send_angles(bots: dict[np.int64,TfBot], player_input_messages: Queue):

    message = "angles |"

    # message format:
    # bot_id pitch (y) yaw (x)
    for bot_id,bot in zip (bots.keys(), bots.values()):
        message += (" {0} {1} {2}\n".format(bot_id, bot.pitch, bot.yaw))

    player_input_messages.put(message)
    send_message.set()



def main():
    initialize_paths()

    bots: dict[np.int64,TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()
    
    iteration = 0
    restart_count = 0
    while not end_program.is_set():
        
        #start program
        start_program.wait()    

        # watiging for squirrel to init it self 
        time.sleep(0.25) 
        
        logger.info("iteration: " + str(iteration))
        #request data postion data
        player_input_messages.put("get_position |") #alway end message_type with "|"
        send_message.set()

        logger.debug("Waiting for positions")
        #waiting for positions
        if not received_positions_data.wait(timeout = MAX_DURATION):
            logger.warning("Received position timeout reached, restarting the loop...")
            restart_count += 1
            logger.warning("Restart count: " + str(restart_count))
            continue

        received_positions_data.clear() #removing flag


        # HERE WE PUT OUR GREAT AI MODEL
        for bot in bots.values():
            bot.pitch = random.uniform(-179,179)
            bot.yaw =  random.uniform(-89,89) # yaw < 0  = up

        logger.debug("Sending angles")
        send_angles(bots, player_input_messages) 
        
        # wait for damage response
        time.sleep(3.0)

        #requesting damage data
        player_input_messages.put( "send_damage|" )
        send_message.set()

        
        logger.debug("waiting for damage data")
        # waiting for damage data
        if not received_damage_data.wait(MAX_DURATION):
            logger.warning("Received damage data timeout reached, restarting the loop...")
            restart_count += 1
            logger.warning("Restart count: " + str(restart_count))
            continue
        received_damage_data.clear() # don't forget to clear the flag
        
        logger.info("Bot damage data: ")
        for bot_id, bot in zip(bots.keys(), bots.values()):
            if bot.damage_dealt != 0:
                logger.info("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
 
            #reseting damage_dealt!!
            bot.damage_dealt = 0


        # requesting bullets distances from target_bot
        player_input_messages.put("send_distances|")
        send_message.set()

        logger.debug("waiting for bullet data")
        # waiting for damage data
        if not received_bullet_data.wait(MAX_DURATION):
            logger.warning("Received bullet data timeout reached, restarting the loop...")
            restart_count += 1
            logger.warning("Restart count: " + str(restart_count))
            continue
        received_bullet_data.clear() # don't forget to clear the flag
        
        logger.info("Bot bullet data: ")
        for bot_id, bot in zip(bots.keys(), bots.values()):
            if bot.bullet_distance != 0:
                logger.info("Bot: {0} , distance: {1}".format(bot_id,bot.bullet_distance))
 
            #reseting bullet_distance!!
            bot.bullet_distance = 0

        iteration += 1
        #loop ends ig
        
    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()