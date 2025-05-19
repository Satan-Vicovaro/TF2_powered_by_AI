import random
import threading
import time
import os
from queue import Queue
import numpy as np
from enum import Enum, auto


#replace those lines with proper paths:
SQUIRREL_IN_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_in" 
SQUIRREL_OUT_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_out"
POLLING_INTERVAL = 0.01 # in seconds [s]


end_program = threading.Event()

start_program = threading.Event()

send_message = threading.Event()

received_positions_data = threading.Event()

received_damage_data = threading.Event()

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
            damage_dealt: float = 0.0
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





def handle_squirrel_input(player_input_messages:Queue):
    if player_input_messages.empty() :     
        return
    
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
          
        lines = in_file.readlines()

        #removing last character of last line (this buggy nullbyte)
        if lines and lines[-1]:  
            lines[-1] = lines[-1][:-1]

        if lines[0] == "none" :
            received_damage_data.set()
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
                        bots[bot_id] = TfBot(pos_x = pos_x, pos_y = pos_y,pos_z = pos_z, bot_type = bot_type)

                case"d": 
                    damage_data = True
                    bot_id = np.int64(parts[1])
                    damage = np.float64(parts[2])
                    if not bot_id in bots:
                        print("Error: Bot with id: " + str(bot_id) + " does not exist")                    
                    bots[bot_id].damage_dealt = damage

                case"_":
                    print("Error: why start of message is: " + parts[0])


        if damage_data and position_data:
            print("Error? : I got damage_data and postion_data in 1 message")
        elif damage_data:
            received_damage_data.set()
        elif position_data:
            received_positions_data.set()

        in_file.truncate(0) # clearing contents
    

def tf2_listener_and_sender(player_input_messages:Queue, bots:dict[np.int64,TfBot]):
    print("Listening on tf2 files")

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
    print("listening for player input") 
    while not end_program.is_set():
        try:
            user_input = input("[You] > ").strip()
            if user_input.lower() == "exit":
                end_program.set()
            elif user_input.lower() == "start":
                player_input_messages.put("start |")
                start_program.set()
            else:
                player_input_messages.put(user_input + " |")  

            send_message.set()          
                                
        except (KeyboardInterrupt,EOFError):
            print("Keyboard error occured")
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
    
    bots: dict[np.int64,TfBot] = {}
    
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = tf2_listener_and_sender, args = (player_input_messages,bots, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()

    iteration  = 0.0

    while not end_program.is_set():
        #start program
        if not start_program.is_set():
            time.sleep(0.05)
            continue
        
        time.sleep(0.25)
        
        #request data postion data
        player_input_messages.put("get_position |") #alway end message_type with "|"
        send_message.set()

        #waiting for response
        while not received_positions_data.is_set():
            
            if end_program.is_set():        
                tf_listener.join()
                user_listener.join()
                return
            
            time.sleep(0.25)
        
        received_positions_data.clear() #removing flag

        # HERE WE PUT OUR GREAT AI MODEL
        for bot in bots.values():
            bot.pitch = random.uniform(-179,179)

            bot.yaw =  random.uniform(-89,89) # yaw (-89,89?) -89 = up

        
        #print("Sending angles")
        send_angles(bots, player_input_messages) 
        
        # wait for damage response
        time.sleep(3.0)

        player_input_messages.put( "send_damage|" )
        send_message.set()

        while not received_damage_data.is_set(): #and not end_program.is_set():
            if end_program.is_set():        
                tf_listener.join()
                user_listener.join()
                return
            time.sleep(0.01)
        received_damage_data.clear()

        #evaluate function        
        if not received_damage_data.is_set():
            print("Nobody hit the targed")
       
        for bot_id, bot in zip(bots.keys(), bots.values()):
            if bot.damage_dealt != 0:
                print("Bot: {0} , dealt: {1}".format(bot_id,bot.damage_dealt))
            #else:
            #   print("Bot: {0} , did not hit".format(bot_id))
            
            #reseting damage_dealt!!
            bot.damage_dealt = 0

        #loop ends ig
        #iteration += 1


    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()