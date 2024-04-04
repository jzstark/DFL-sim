import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import libsumo as traci
import traci.constants as tc

sumoBinary = "/usr/bin/sumo"
sumoCmd = [sumoBinary, "-c", "manhattan/data/manhattan.sumocfg"]

traci.start(sumoCmd)
step = 0

traci.simulationStep()
vehIDList = traci.vehicle.getIDList()




for vehID in vehIDList:
    traci.vehicle.subscribe(vehID, (tc.POSITION_2D, tc.VAR_POSITION, tc.VAR_SPEED,
                                        tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))

traci.simulationStep()


while step < 10:
    traci.simulationStep()
    print(step, traci.simulation.getTime(), traci.simulation.getCurrentTime())
    # print(traci.vehicle.getIDList())
    print(traci.vehicle.getSubscriptionResults('1.0'))
    step += 1

traci.close()