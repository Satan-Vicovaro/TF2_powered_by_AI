# TF2_Powered_by_AI

This project aims to implement a neural network that attempts to hit a target within the Team Fortress 2 (TF2) game environment. It was created as part of the *Artificial Intelligence* course at Gdańsk University of Technology.




## Features

### TF2 Vscripts
- Bot position tracking  
- Soldier's missile tracking  
- Bot spawning in designated positions  
- Python listener integration

### Python Components
- Communication interface with TF2  
- Two neural network implementations (including a DDPG model)  
- Basic logging system

> **Note:** The neural networks are currently non-functional, likely due to bugs in the code. This might be resolved in the future.

The DDPG implementation is based on this:  
https://www.kaggle.com/code/auxeno/ddpg-rl/notebook


---

## Getting Started

### Prerequisites
- Python 3
- A working installation of Team Fortress 2 

### Setup Instructions

1. **Initial Configuration**
 ```
 python load_to_tf2.py
 ```
- This creates a `config.json` file. Open it and specify the correct paths to your directories at TF2.

2. **Load Files into TF2**

    Run this command again to load the necessary files into the appropriate TF2 directories:
```
python load_to_tf2.py
```

4. **Start TF2 and Create a Game**
  - Launch Team Fortress 2.
  - Start a new game on map: `proper_train_map`.

4. **Initialize the Server**
  - Open the in-game developer console.
  - Run the following command:
```
exec server_init
```
5. **Start the AI Script**
```
python ai_script.py
```
  - Then type: `start` in the terminal to begin the AI routine.

---
## Useful Resources

- https://developer.valvesoftware.com/wiki/Team_Fortress_2/Scripting/Script_Functions  
- https://developer.valvesoftware.com/wiki/Team_Fortress_2/Scripting/VScript_Examples  
- https://developer.valvesoftware.com/wiki/Team_Fortress_2/Scripting/Script_Functions/Constants  
- https://github.com/ValveSoftware/source-sdk-2013





