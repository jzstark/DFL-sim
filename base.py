import numpy as np
from typing import Type, Dict, Any
import sys
import random
from queue import PriorityQueue
import heapq
from collections import defaultdict


class PriorityQueue:
    def __init__(self, lst : list = []) -> None:
        self.heap = lst
        self.count = len(lst)
        heapq.heapify(self.heap)

    def push(self, item):
        heapq.heappush(self.heap, item)
        self.count += 1
    
    def pop(self):
        if self.count > 0:
            self.count -= 1
            return heapq.heappop(self.heap)
        raise IndexError("Pop from empty queue")
    
    def qsize(self):
        return self.count 


class Message:
    def __init__(self, content, delivery_time=0) -> None:
        self.content = content
        self.delivery_time = delivery_time

        self.error = 0 # error rate
        self.speed = 1 # Mbps? 

    def __lt__(self, other: 'Message'):
        return self.delivery_time < other.delivery_time


class Channel: 
    def __init__(self, drop_rate = 0, bandwidth = 1) -> None:
        # LTE based C-V2X can reach 10s of Mbps, and 5G can achieve even Gbps, ideally
        self.drop_rate = drop_rate
        self.bandwidth = bandwidth


# Each agent represent a single vehicle
class BaseAgent():
    
    def __init__(self, vehID : str, chan : Channel) -> None:
        self.id = vehID
        self.data = None
        self.position : tuple[float, float] = (0, 0)
        # Data Collected from different sources, at different time slot
        # self.cache : Dict[str, PriorityQueue[Message]] = {}
        self.cache : defaultdict[str, Message] = defaultdict(str)
        self.channel = chan
        self.time = 0
        # Compute time : random [0-1] time unit
        
    
    def set_position(self, pos : tuple[float, float]):
        self.position = pos

    
    def set_cache(self, src_id : str,  msg : Message) -> None:
        #queue = self.cache[src_id]
        #queue.push(msg)
        self.cache[src_id] = msg

    def flat_cached_data(self) :
        data = []
        for msg in self.cache.values():
            data.append(msg.content)
        return data
    
    def set_time(self, time) :
        self.time = time 

    def send(self, msg, dest: 'BaseAgent') -> None:
        #TODO: perhaps very heavy operations:
        if random.random() < self.channel.drop_rate:
            return
        # delivery_time = sys.getsizeof(msg) / self.channel.bandwidth
        # TODO: what is suitable delivery_time ? Currently assume no time delay
        delivery_time = 0
        msg = Message(msg, delivery_time=self.time + delivery_time)
        dest.set_cache(self.id, msg)

    def get_data(self):
        pass

    def get_diff_data(self):
        pass

    def aggregate(self):
        pass
            
    def updateLocalData(self):
        pass 

    #NOTE: assume timelen is normally 1 or several "time unit"
    def localComputeUntil(self, timelen):
        start_time = 0
        while (start_time < timelen):
            self.updateLocalData()
            #TODO: update start_time with some computing time
            start_time += random.random()


    # def duplicateMsg(self, ms : list[Message]):
    #     pass
    #def msg_status(self, src, dest):
    #    return self.msg_matrix[src][dest]