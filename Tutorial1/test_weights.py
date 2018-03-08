import numpy as np
from Simulation import Simulation

env_range = 20

pos = np.array([-6,-6])
# Defin robot radius, sensor range, 1/dT for how many times to render simulation within one loop of robot controller
robot_rad = 1
sens_range = 3
dT = 0.1
np.random.seed(5)
iter_sim = 1000
#Define datetime of directory with saved weights
datetime = '1520520627'

sim = Simulation(iter_sim, env_range, pos, robot_rad, sens_range, dT, graphics = True)

def tokenizer(fname):
    with open(fname) as f:
            weights= []
            for line in f:
                    if 'end_layer' in line:
                            yield weights                            
                            weights = []
                            continue
                    weights.append(line)
                    
weights = np.array([np.loadtxt(A,delimiter=',') for A in tokenizer('./output/'+datetime+'/weights.txt')])

sim.simulate(weights)