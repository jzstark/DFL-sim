#!/usr/bin/python
from typing import Type
from base import BaseAgent

def bsp(a : BaseAgent, nodeID, condition : function):
    # get the 
    neighbors = a.filter(condition)
    # if all neighbors are "ready": no sent-but-not-received 
    progress = [a.msg_status(n, nodeID) for n in neighbors]
    return all(progress) 

def asp(a, nodeID, condition):
    return True
    
