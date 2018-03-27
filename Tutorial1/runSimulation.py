import numpy as np
from Simulation import Simulation
'''
Run this script to test controller in simulation environment with localization filter.
Localization filter is included by flag 'localiz=1', either kalmanFilter() 
or extendedKalmanFilter can be selected in simulation script.
'''

# Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
env_range = 20
robot_rad = 1
sens_range = 3
dT = 1/3
iter_sim = 1000

def tokenizer(fname):
    with open(fname) as f:
            weights= []
            for line in f:
                    if 'end_layer' in line:
                            yield weights                            
                            weights = []
                            continue
                    weights.append(line)
                    
#Test on Map 1
# sim_map = np.load('Maps/'+'Map1'+'.npy')
# locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6]])
# pos = locations[np.random.randint(locations.shape[0])]   
# weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map1.txt')])
# sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
# sim.simulate(weights, sim_map, pos, localiz=1)

#Test on Map 2
# sim_map = np.load('Maps/'+'Map2'+'.npy')
# locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
# pos = locations[np.random.randint(locations.shape[0])]   
# weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map2.txt')])
# sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
# sim.simulate(weights, sim_map, pos, localiz=1)

# Test on Map 4
sim_map = np.load('Maps/'+'Map4'+'.npy')
locations = np.array([[-6,-6],[6,-6],[6,6],[-6,6],[0,0]])
pos = locations[np.random.randint(locations.shape[0])]   
# pos = np.array([6,6])
weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('weights_map4.txt')])
sim = Simulation(iter_sim, env_range, robot_rad, sens_range, dT, None, graphics = True)
sim.simulate(weights, sim_map, pos,localiz=1)






	

