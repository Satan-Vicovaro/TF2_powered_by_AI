import threading
import time
import logger as lg
import globals as gl
from queue import Queue
from data_collector import shared_collector, DataCollector


class UserListener:

    def __init__(self) -> None:
        self.user_input_handler = threading.Thread(
            target=self.user_input_listener, args=(gl.player_input_messages,), daemon=True
        )

    def start(self):
        self.user_input_handler.start()

    def stop(self):
        self.user_input_handler.join()

    def user_input_listener(self, player_input_messages: Queue):
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
                elif user_input.lower() == "load nn":
                    lg.logger.info("Neural network will be loaded from file")
                    gl.load_neural_network = True
                    continue
                elif user_input.lower() == "debug on":
                    lg.enable_debug()
                elif user_input.lower() == "debug off":
                    lg.disable_debug()
                elif user_input.lower() == "plot data":
                    # warining this might fail ploting should be on main thread
                    threading.Thread(
                        target=lambda: shared_collector.plot_data(), daemon=True
                    ).start()
                elif user_input.lower() == "plot file":
                    # warining this might fail ploting should be on main thread
                    def thread_task():
                        new_collector = DataCollector()
                        new_collector.load_data()
                        new_collector.plot_data()

                    threading.Thread(target=thread_task(), daemon=True).start()
                elif user_input.lower() == "dummy":
                    lg.logger.info("Set dummy env")
                    gl.enviroment_type = "dummy"
                elif user_input.lower() == "normal":
                    lg.logger.info("Set normal env")
                    gl.enviroment_type = "normal"
                elif user_input.lower() == "help":
                    lg.logger.info(
                        "Options: start \n load nn \n debug on \n debug off \n plot data \n plot file \n dummy \n normal \n help \n"
                    )
                elif user_input.lower() == "act":
                    lg.logger.info("Bot will not be learning")
                    gl.is_learning = False
                else:
                    player_input_messages.put(user_input + " |")

                lg.logger.debug("User_listener: Got input form player")
                gl.send_message.set()

            except (KeyboardInterrupt, EOFError):
                lg.logger.error("Keyboard error occured")
                gl.end_program.set()
