import random
import threading
import time
import os
from queue import Queue

#replace those lines with proper paths:
SQUIRREL_IN_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_in" 
SQUIRREL_OUT_PATH = "/mnt/a3e71cc8-0490-43a5-abb9-3d05162d3dee/SteamLibrary/steamapps/common/Team Fortress 2/tf/scriptdata/squirrel_out"
POLLING_INTERVAL = 0.01 # in seconds [s]


end_program = threading.Event()

send_message = threading.Event()

def tf2_listener_and_sender(player_input_messages:Queue):
    print("Listening on tf2 files")

    while not end_program.is_set():
        
        #if file is not empty
        if os.stat(SQUIRREL_OUT_PATH).st_size > 1 : # file is empty or have one null byte (Squirrel bug) 
            #looking at IN file
            with open(SQUIRREL_OUT_PATH, 'r+') as in_file: #squirrel_out == python_in 
                lines = in_file.readlines()
            
                i = 0
                for line in lines:
                    print(str(i) + ": " + repr(line))
                    i += 1
        
                in_file.truncate(0) # clearing contents

            
            if not send_message.is_set():
                time.sleep(POLLING_INTERVAL)
                continue
            
        #if file is not empty
        if not player_input_messages.empty() :     
            #looking at OUT file
            with open(SQUIRREL_IN_PATH, "w+") as out_file: #squirrel_in == python_out, a+ == append + read
                
                while(not player_input_messages.empty()):
                    out_file.write(player_input_messages.get() + "\n")
                 
        
        time.sleep(POLLING_INTERVAL)
    
    #sending exit command to python_listener in tf2
    with open(SQUIRREL_IN_PATH, "w") as out_file:
        out_file.write("exit")

def user_input_listener(player_input_messages:Queue):
    print("listening for player input") 
    while not end_program.is_set():
        try:
            user_input = input("[You] > ").strip()
            if user_input.lower() == "exit":
                end_program.set()
            else:
                player_input_messages.put(user_input)            
                                
        except (KeyboardInterrupt,EOFError):
            print("Keyboard error occured")
            end_program.set()




def main():
    player_input_messages = Queue(2137)  
    
    tf_listener = threading.Thread(target = tf2_listener_and_sender, args = (player_input_messages, ), daemon = True)
    user_listener = threading.Thread(target = user_input_listener, args = (player_input_messages, ), daemon = True)
    
    tf_listener.start()
    user_listener.start()

    # while not end_program.is_set():
    #     message = ""
    #     for i in range(0,100):
    #         message += " {0} {1} {2}\n".format(
    #             i,
    #             random.random(),
    #             random.random())

    #     #print(message + "\n")
    #     player_input_messages.put(message)
    #     time.sleep(0.05)


    tf_listener.join()
    user_listener.join()

    return


if __name__ == "__main__":
    main()