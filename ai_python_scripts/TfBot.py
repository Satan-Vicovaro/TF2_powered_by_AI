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
            self.pos_x = np.float64(pos_x)
            self.pos_y = np.float64(pos_y)
            self.pos_z = np.float64(pos_z)

            self.pitch = np.float64(pitch)
            self.yaw = np.float64(yaw)

            self.vel_x = np.float64(vel_x)
            self.vel_y = np.float64(vel_y)
            self.vel_z = np.float64(vel_z)

            self.bot_type = bot_type
            self.damage_dealt = np.float32(damage_dealt)

            self.m_miss_x = np.float32(m_miss_x)
            self.m_miss_y = np.float32(m_miss_y)
            self.m_miss_z = np.float32(m_miss_z)

            self.m_distance = np.float32(m_distance)

            self.m_x = np.float32(m_x)
            self.m_y = np.float32(m_y)
            self.m_z = np.float32(m_z)


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
        self.m_z /= normalize_factor        self.m_x /= normalize_factor
        self.m_y /= normalize_factor
        self.m_z /= normalize_factor

