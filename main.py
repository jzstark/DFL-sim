import os
import sys
import pickle
import matplotlib.pyplot as plt


if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import libsumo as traci
import traci.constants as tc

from sim import Simulation
from svdAgent import SVDAgent
from base import Channel
from simpleFLAgent import SimpleDNNAgent

simple_chan = Channel()
simulation = Simulation([], SimpleDNNAgent, simple_chan, ob_interval=1)
#simulation = Simulation([], SVDAgent, simple_chan, ob_interval=1)

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

means = []
stds  = []
while step < 50:
    traci.simulationStep()
    # print(step, traci.simulation.getTime(), traci.simulation.getCurrentTime())
    # print(traci.vehicle.getIDList())
    for vid in vehIDList:
        subscription = traci.vehicle.getSubscriptionResults(vid)
        simulation.update_agent_position(
            vid, subscription[tc.VAR_POSITION])
        
    simulation.step_async()
    if step % 2 == 0:
        #mu, std = simulation.test_svd()
        mu, std = simulation.test_acc_dnn()
        print(mu, std)
        means.append(mu)
        stds.append(std)
    
    step += 1
    simulation.set_time(step)

traci.close()


plt.errorbar(range(len(means)), means, yerr=stds, capsize=5, ecolor='red', 
             linestyle='-', marker='o', color='blue', label='Average with Std Dev')

# Adding labels and title for clarity
plt.xlabel('Iterations')
plt.ylabel('Cosine distances to original matrix')
plt.title('Decentralized SVD on MovieLens-100K dataset')
plt.savefig('test.png')


with open('mean.pkl', 'wb') as f:
    pickle.dump(means, f)

with open('std.pkl', 'wb') as f:
    pickle.dump(stds, f)

"""
with open('mean.pkl', 'wb') as f:
    means = pickle.load(means, f)

with open('std.pkl', 'wb') as f:
    stds = pickle.load(stds, f)
"""