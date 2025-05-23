import threading 

#------------------------------------ thread flags -----------------------------------
end_program = threading.Event()

start_program = threading.Event()

send_message = threading.Event()

received_positions_data = threading.Event()

received_damage_data = threading.Event()


POLLING_INTERVAL = 0.01 # in seconds [s]
MAX_DURATION = 10 # seconds