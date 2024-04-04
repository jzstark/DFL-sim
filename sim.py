
from base import BaseAgent, Channel
from typing import Dict
import numpy as np 

# We use a unified time, controlled by simulator

class Simulation : 
    def __init__(self, vehIDs : list[str], agentType : BaseAgent, 
                 chan : Channel, ob_interval : float) -> None:
        self.agentType = agentType
        self.agents : Dict[str, BaseAgent]= {} 
        self.channel = chan
        self.ob_interval = ob_interval
        self.time = 0 # global time 

    def add_vehicle(self, vehID, position = (0,0)) -> None:
        v : BaseAgent = self.agentType(vehID, self.channel)
        v.set_position(position)
        self.agents[vehID] = v
        
    def get_time(self):
        return self.time

    def set_time(self, t):
        self.time = t


    def update_agent_position(self, vid : str, pos : tuple[float, float]):
        a = self.agents[vid]
        a.set_position(pos)


    def step_async(self):
        # Assume the positions of agents are updated
        for a in self.agents.values():
            a.set_time(self.time)
        
        for a in self.agents.values():
            #TODO: find a suitable group of neighbors
            group : list[BaseAgent] = []
            
            # "Compute your local weights iteratively within this step"
            for g in group:
                g.localComputeUntil(self.ob_interval)
            
            # "send your local weights to me"          
            for g in group:
                g.send(g.get_data(), a)
            
            # Consensus of data
            a.aggregate()

        return
    
"""
 #def _check_sender(self, num) -> list[int]:
    #    assert(num >=0 and num < self.n)
    #    c = self.comm_matrix[:, num]
    #    indices = np.where(c > 0)[0]
    #    return list(indices)


self.id2num : Dict[str, int] = {}
        self.num2id : Dict[int, str] = {}
        counter = 0
        for id in vehIDs:
            self.agents[id] = agentType(id, chan)
            self.id2num[id] = counter 
            self.num2id[counter] = id 
            counter += 1
        #NOTE: assumption is that the number of vehicles are fixed 
        self.n = len(vehIDs)
        # row: sender; column: receiver 
        self.comm_matrix = np.zeros([self.n, self.n])


# if I've received a message ... 
            # check the matrix to see who is sending to me 
            senders = self._check_sender(self.id2num[a.id])
            for i in senders:
                # check each sender's list of sent message, take all that is usable (time already passed), 
                # apply computation|
                sender_id = self.num2id[i]
                senderMsg = a.duplicateMsg(a.popValidMsgFrom(
                    sender_id, current_time=self.time))
                
            
            # if the barrier passes, I'm OK to move on  to compute 
            if a.barrier(group) : 
                #TODO: the computation also takes time 
                a.compute()

                # After computing 
                for b in group:
                    a.send(a.get_data(), b)
"""