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
            m_miss_x: float = 0.0,
            m_miss_y:float = 0.0,
            m_miss_z: float = 0.0,
            m_distance: float = 0.0,
            m_x: float = 0.0,
            m_y: float = 0.0,
            m_z: float = 0.0
        ):
            self.pos_x = float(pos_x)
            self.pos_y = float(pos_y)
            self.pos_z = float(pos_z)

            self.pitch = float(pitch)
            self.yaw = float(yaw)

            self.vel_x = float(vel_x)
            self.vel_y = float(vel_y)
            self.vel_z = float(vel_z)

            self.bot_type = bot_type
            self.damage_dealt = float(damage_dealt)

            self.m_miss_x = float(m_miss_x)
            self.m_miss_y = float(m_miss_y)
            self.m_miss_z = float(m_miss_z)

            self.m_distance = float(m_distance)

            self.m_x = float(m_x)
            self.m_y = float(m_y)
            self.m_z = float(m_z)


    def normalize(self):        
        normalize_factor = 1000.0 #aka radius of our circle
        self.pos_x /= normalize_factor
        self.pos_y /= normalize_factor
        self.pos_z /= normalize_factor

    def normalize_missiles(self):
        normalize_factor = 1000.0 #aka radius of our circle
        self.m_miss_x /= normalize_factor
        self.m_miss_y /= normalize_factor
        self.m_miss_z /= normalize_factor
        
        self.m_x /= normalize_factor
        self.m_y /= normalize_factor
        self.m_z /= normalize_factor
        

