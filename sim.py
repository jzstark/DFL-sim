
from base import BaseAgent
from typing import Dict
import numpy as np 

class Simulation : 
    def __init__(self, vehIDs : list[str], agentType : BaseAgent, barrier) -> None:
        self.agents : Dict[str, BaseAgent]= {} 
        self.id2num : Dict[str, int] = {}
        self.num2id : Dict[int, str] = {}
        counter = 0
        for id in vehIDs:
            self.agents[id] = BaseAgent(id)
            self.id2num[id] = counter 
            self.num2id[counter] = id 
            counter += 1
        
        self.n = len(vehIDs)
        # row: sender; column: receiver 
        self.comm_matrix = np.zeros([self.n, self.n])
        self.time = 0


    def get_time(self):
        return self.time

    def set_time(self, t):
        self.time = t

    def _check_sender(self, num) -> list[int]:
        assert(num >=0 and num < self.n)
        c = self.comm_matrix[:, num]
        indices = np.where(c > 0)[0]
        return list(indices)

    def step(self):
        # at each step, each agent runs "in parallel": 
        for a in self.agents.values():
            #!!! find a suitable group of a 
            group : list[BaseAgent] = []
            
            # if I can send message: process is over, send it 
            if not a.isComputing(): 
                for b in group:
                    a.send(a.data, b) 
            
            # if I've received a message ... 
            # check the matrix to see who is sending to me 
            senders = self._check_sender(self.id2num[a.id])
            for i in senders:
                a.receive()
            
            # if the barrier passes, I'm OK to move on  to compute 
            if a.barrier(group) : 
                a.compute()
        
        return