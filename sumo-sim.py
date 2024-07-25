import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set global font size
plt.rcParams.update({'font.size': 14})

"""
from scipy.spatial import KDTree

class AgentKDTree:
    def __init__(self, agents : Dict[str, BaseAgent]) -> None:
        self.agents = agents
        id_points = [(a.id, a.position) for a in self.agents.values()]
        self.vid, points = zip(*id_points)
        self.tree = KDTree(points)
    
    def find_neighbors(self, point : tuple[float, float], 
                       range : float) -> list[str] :
        indices = self.tree.query_ball_point(point, range)
        return [self.vid[i] for i in indices]
"""


if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = '/usr/share/sumo/'
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import libsumo as traci
import traci.constants as tc

sumoBinary = "/usr/share/sumo/bin/sumo"
#sumoCmd = [sumoBinary, "-c", "maps/manhattan/data/manhattan.sumocfg"]
sumoCmd = [sumoBinary, "-c", "maps/berlin/berlin.sumocfg"]


traci.start(sumoCmd)
step = 0

traci.simulationStep()

vehIDList = traci.vehicle.getIDList()
for vehID in vehIDList:
    # traci.vehicle.subscribe(vehID, (tc.POSITION_2D, tc.VAR_POSITION, tc.VAR_SPEED,
    #                                    tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
    traci.vehicle.subscribe(vehID, [tc.VAR_POSITION])
    # Output format: {66: (201.6, 891.7949078900516)}


pos_dict = {}

radius = 200 # (300-1000m max for C-V2X)

def distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def approximate_length(lst1, lst2, tolerance=0):
    set1 = set(lst1)
    set2 = set(lst2)
    different_elements = set1.symmetric_difference(set2)
    if len(different_elements) <= tolerance: 
        #print("yes!", len(lst1))
        # if no change (approximately), return the "old" length
        return len(lst1)
    #print("no!", len(lst2))
    return 0


T = 100 

tolerence_num = 3 # 0, 1, 2

results = [np.zeros([tolerence_num, T]), np.zeros([tolerence_num, T])]
neighbors = [[], []]

for step in range(T):
    traci.simulationStep()
    print(step)
    for vid in vehIDList:
        subscription = traci.vehicle.getSubscriptionResults(vid)
        if subscription == {}:
            pos_dict[vid] = (10000000000000, 0)
        else:
            pos_dict[vid] = subscription[tc.VAR_POSITION]
    
    neighbors_of_0 = [vid for vid in vehIDList if distance(pos_dict[vehIDList[1]], pos_dict[vid]) <= radius]
    neighbors_of_1 = [vid for vid in vehIDList if distance(pos_dict[vehIDList[9]], pos_dict[vid]) <= radius]

    neighbors[0].append(neighbors_of_0)
    neighbors[1].append(neighbors_of_1)

    print(neighbors_of_0)
    print(neighbors_of_1)
    
    if step == 0:
        stable_nei_len_0 = len(neighbors_of_0) - 1 # remove itself 
        stable_nei_len_1 = len(neighbors_of_1) - 1 # remove itself 
        for ii in range(tolerence_num): results[0][ii, 0] = stable_nei_len_0
        for ii in range(tolerence_num): results[1][ii, 0] = stable_nei_len_1
    else:
        results[0][0, step ] = approximate_length(neighbors[0][step - 1], neighbors[0][step], tolerance = 0)
        results[0][1, step ] = approximate_length(neighbors[0][step - 1], neighbors[0][step], tolerance = 1)
        results[0][2, step ] = approximate_length(neighbors[0][step - 1], neighbors[0][step], tolerance = 2)

        results[1][0, step ] = approximate_length(neighbors[1][step - 1], neighbors[1][step], tolerance = 0)
        results[1][1, step ] = approximate_length(neighbors[1][step - 1], neighbors[1][step], tolerance = 1)
        results[1][2, step ] = approximate_length(neighbors[1][step - 1], neighbors[1][step], tolerance = 2)

traci.close()

print(results)

fig, axes = plt.subplots(1,2, figsize=(12, 5))

axes[0].plot(list(range(T)), results[0][0], label="tol=0")
axes[0].plot(list(range(T)), results[0][1], label="tol=1", linestyle='--')
#axes[0].plot(list(range(T)), results[0][2], label="tol=2")
axes[0].legend()
axes[0].set_ylabel("Group Size (#)")
axes[0].set_xlabel("Time (s)")

axes[1].plot(list(range(T)), results[1][0], label="tol=0")
axes[1].plot(list(range(T)), results[1][1], label="tol=1",linestyle='--')
#axes[1].plot(list(range(T)), results[1][2], label="tol=2")
axes[1].legend()
axes[1].set_xlabel("Time s)")


plt.show()