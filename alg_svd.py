from base import BaseAgent, Message, Channel
import random
import math

class AlgSVD(BaseAgent) : 
    def __init__(self, vehID: str, chan: Channel) -> None:
        self.a = None
        self.x = None
        self.Y = None 
        super().__init__(vehID, chan)
    
    def get_data(self):
        return (self.a, self.X, self.Y)
    
    def updateLocalData(self):
        pass
    

class SimpleAgent(BaseAgent):
    def __init__(self, vehID: str, chan: Channel) -> None:
        super().__init__(vehID, chan) 
        self.v : float = 0.
        self.old_v: float = 0
    
    def get_data(self) -> Message:
        return self.v
    
    def get_diff_data(self):
        return self.old_v
    
    def updateLocalData(self):
        self.old_v = self.v 
        self.v += random.random()
    
    def aggregate(self):
        s = 0
        c = 1
        for d in self.flat_cached_data():
            s += d
            c += 1
        self.v = s / c