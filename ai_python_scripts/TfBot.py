from enum import Enum, auto
import numpy as np

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
            self.pos_x = np.float32(pos_x)
            self.pos_y = np.float32(pos_y)
            self.pos_z = np.float32(pos_z)

            self.pitch = np.float32(pitch)
            self.yaw = np.float32(yaw)

            self.vel_x = np.float32(vel_x)
            self.vel_y = np.float32(vel_y)
            self.vel_z = np.float32(vel_z)

            self.bot_type = bot_type
            self.damage_dealt = np.float32(damage_dealt) 
            self.bullet_distance = np.float32(bullet_distance)

    def normalize(self):        
        normalize_factor = 1000.0 #aka radius of out circle
        self.pos_x /= normalize_factor
        self.pos_y /= normalize_factor
        self.pos_z /= normalize_factor
        

