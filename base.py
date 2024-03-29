import numpy as np
from typing import Type, Dict, Any
import time


class Channel: 
    def __init__(self, error = 0) -> None:
        self.error = 0 # error rate
        self.speed = 1 # Mbps? 
    
    def send(self, msg, receiver: 'BaseAgent'):
        time.sleep(0) 
        # some other stuff
        receiver.receive(msg)


class BaseAgent():
    
    def __init__(self, vehID : str, channel: Channel) -> None:
        self.id = vehID
        self.data = None
        self.compute_lock = False
        self.position = (0, 0)
        self.cache : Dict[str, list[Any]] = {}
        self.channel = channel

    def get_data(self):
        return self.data
    
    def set_data(self, d) -> None:
        self.data = d
    
    def isComputing(self) -> bool:
        return self.compute_lock

    def send(self, msg, dest: 'BaseAgent') -> None:
        # should specify some condition
        # so that even
        # NOTE: error here 
        lst = dest.cache[dest.id]
        lst.add(msg)
        
    def receive(self, msg) -> None:
        # should check if the condition is met
        self.data = msg

    def compute(self):
        pass 
    
    #def msg_status(self, src, dest):
    #    return self.msg_matrix[src][dest]