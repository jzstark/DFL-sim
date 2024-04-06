import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import libsumo as traci
import traci.constants as tc

from sim import Simulation
from alg_svd import SimpleAgent
from base import Channel
from simpleFLAgent import SimpleDNNAgent

simple_chan = Channel()
#simulation = Simulation([], SimpleAgent, simple_chan, ob_interval=1)
simulation = Simulation([], SimpleDNNAgent, simple_chan, ob_interval=1)

sumoBinary = "/usr/bin/sumo"
sumoCmd = [sumoBinary, "-c", "manhattan/data/manhattan.sumocfg"]

traci.start(sumoCmd)
step = 0

traci.simulationStep()

vehIDList = traci.vehicle.getIDList()
for vehID in vehIDList:
    # traci.vehicle.subscribe(vehID, (tc.POSITION_2D, tc.VAR_POSITION, tc.VAR_SPEED,
    #                                    tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
    traci.vehicle.subscribe(vehID, [tc.VAR_POSITION])
    # Output format: {66: (201.6, 891.7949078900516)}
    simulation.add_vehicle(vehID)

while step < 10:
    traci.simulationStep()
    # print(step, traci.simulation.getTime(), traci.simulation.getCurrentTime())
    # print(traci.vehicle.getIDList())
    for vid in vehIDList:
        subscription = traci.vehicle.getSubscriptionResults(vid)
        simulation.update_agent_position(
            vid, subscription[tc.VAR_POSITION])
        
    simulation.step_async()
    #if step % 2 == 0:
    print(simulation.test_acc())
    
    step += 1
    simulation.set_time(step)

traci.close()